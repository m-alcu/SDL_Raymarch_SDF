// Dear ImGui: standalone example application for GLFW + OpenGL 3, using programmable pipeline
// (GLFW is a cross-platform general purpose library for handling windows, inputs, OpenGL/Vulkan/Metal graphics context creation, etc.)

// Learn about Dear ImGui:
// - FAQ                  https://dearimgui.com/faq
// - Getting Started      https://dearimgui.com/getting-started
// - Documentation        https://dearimgui.com/docs (same as your local docs/ folder).
// - Introduction, links and more at the top of imgui.cpp

#include "vendor/imgui/imgui.h"
#include "vendor/imgui/imgui_impl_glfw.h"
#include "vendor/imgui/imgui_impl_opengl3.h"
#include <stdio.h>
#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
// Expose prototypes for all GL 2.x/3.x functions from the system libGL.so
// (works on Linux/Mesa; the Makefile already links -lGL)
#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h> // Will drag system OpenGL headers
#include <GL/glext.h>

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and compatibility with old VS compilers.
// To link with VS2010-era libraries, VS2015+ requires linking with legacy_stdio_definitions.lib, which we do using this pragma.
// Your own project should not be affected, as you are likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

// This example can also compile and run with Emscripten! See 'Makefile.emscripten' for details.
#ifdef __EMSCRIPTEN__
#include "../libs/emscripten/emscripten_mainloop_stub.h"
#endif

// -----------------------------------------------------------------------------
// Fullscreen vertex shader
// -----------------------------------------------------------------------------
static const char* g_vert_src = R"GLSL(
#version 130
in vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
)GLSL";

// -----------------------------------------------------------------------------
// The MIT License
// Copyright © 2013 Inigo Quilez
// A list of useful distance functions to simple primitives.
// Adapted from Shadertoy for standalone GLSL 1.30:
//   - iTime, iMouse, iResolution, iFrame are uniforms
//   - mainImage(out vec4, in vec2) is called from main()
// -----------------------------------------------------------------------------
static const char* g_frag_src = R"GLSL(
#version 130

uniform float iTime;
uniform vec2  iMouse;
uniform vec3  iResolution;
uniform int   iFrame;

out vec4 outColor;

// AA = number of subpixel samples per axis (AA*AA total samples per pixel).
// Set HW_PERFORMANCE=0 to disable AA and get 1 sample/pixel for slow GPUs.
#define HW_PERFORMANCE 1

#if HW_PERFORMANCE==0
#define AA 1
#else
#define AA 2   // 2x2 = 4 samples/pixel: grid supersampling, not random (no Monte Carlo noise)
#endif

// --- Utility helpers ---------------------------------------------------------
// dot2: squared length. Avoids a sqrt when you only need to compare distances.
float dot2( in vec2 v ) { return dot(v,v); }
float dot2( in vec3 v ) { return dot(v,v); }
// ndot: "diagonal dot" used by rhombus SDF. Projects onto the rhombus diagonal
// axes instead of the standard axes, enabling the rhombus formula.
float ndot( in vec2 a, in vec2 b ) { return a.x*b.x - a.y*b.y; }

// =============================================================================
// SIGNED DISTANCE FUNCTIONS (SDFs)
// =============================================================================
// An SDF f(p) returns:
//   f(p) < 0  → p is inside the shape
//   f(p) = 0  → p is on the surface
//   f(p) > 0  → p is outside; the value is the exact distance to the surface
//
// This "safe step size" property is what makes sphere tracing possible:
// we can advance the ray by exactly f(p) without overshooting the surface.
// =============================================================================

// PLANE: the simplest SDF. The ground plane at y=0.
// Distance from p to the plane y=0 is simply the y component.
float sdPlane( vec3 p )
{
    return p.y;
}

// SPHERE: distance to surface = distance to center minus radius.
// The most fundamental SDF; all others are derived from this idea.
float sdSphere( vec3 p, float s )
{
    return length(p)-s;
}

// BOX (axis-aligned): uses the "corner trick".
// 1. Fold p into the positive octant with abs(), exploiting 3-axis symmetry.
// 2. d = abs(p) - b gives signed per-axis distances to the box faces.
//    Negative components mean p is inside that slab.
// 3. length(max(d,0)): exterior Euclidean distance (only positive components matter).
// 4. min(max(d.x,d.y,d.z), 0): interior distance (largest negative = closest face).
// Combined, this gives a continuous exact SDF inside and outside.
float sdBox( vec3 p, vec3 b )
{
    vec3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

// BOX FRAME: wireframe box with edge thickness e.
// Trick: instead of carving out the interior, fold p into the corner region
// by applying abs() twice with different offsets. Each of the 3 terms covers
// one pair of parallel edges (x-edges, y-edges, z-edges).
// The min() of the 3 terms gives the closest edge.
float sdBoxFrame( vec3 p, vec3 b, float e )
{
       p = abs(p  )-b;        // fold into positive octant, offset to box face
  vec3 q = abs(p+e)-e;        // fold again at the edge thickness

  // Three terms: each handles the edges running along one axis.
  return min(min(
      length(max(vec3(p.x,q.y,q.z),0.0))+min(max(p.x,max(q.y,q.z)),0.0),
      length(max(vec3(q.x,p.y,q.z),0.0))+min(max(q.x,max(p.y,q.z)),0.0)),
      length(max(vec3(q.x,q.y,p.z),0.0))+min(max(q.x,max(q.y,p.z)),0.0));
}

// ELLIPSOID: approximated (not exact Euclidean distance).
// k0 = normalized length (maps ellipsoid to unit sphere).
// k1 = length after dividing by r^2 (proportional to the gradient magnitude).
// k0*(k0-1)/k1 approximates the true distance by correcting for the non-uniform
// scaling. This overestimates slightly, making sphere tracing conservative (safe).
float sdEllipsoid( in vec3 p, in vec3 r )
{
    float k0 = length(p/r);
    float k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
}

// TORUS: reduce to 2D.
// length(p.xz)-t.x = signed distance from the torus axis circle (the "spine").
// Then distance from that circle to the tube surface = subtract tube radius t.y.
float sdTorus( vec3 p, vec2 t )
{
    return length( vec2(length(p.xz)-t.x,p.y) )-t.y;
}

// CAPPED TORUS: torus with a wedge cut by an opening angle.
// sc = (sin, cos) of the half-angle of the opening.
// Trick: fold the XY plane so the cut is at x=0, then clamp k to the arc.
// When the point is in the angular gap, use projection onto the endpoint cap.
float sdCappedTorus(in vec3 p, in vec2 sc, in float ra, in float rb)
{
    p.x = abs(p.x);   // exploit bilateral symmetry of the opening
    // k = distance along the torus spine, clamped to the arc or capped to the endpoint
    float k = (sc.y*p.x>sc.x*p.y) ? dot(p.xy,sc) : length(p.xy);
    return sqrt( dot(p,p) + ra*ra - 2.0*ra*k ) - rb;
}

// HEX PRISM: computed in 2D cross-section + extrusion.
// The hex cross-section is handled by reflecting p into the fundamental domain
// using the hex lattice normal k.xy = (-sqrt(3)/2, 0.5).
// After two reflections p is in a 30° wedge; then project to nearest edge.
float sdHexPrism( vec3 p, vec2 h )
{
    vec3 q = abs(p);   // unused after re-abs below, but kept for clarity

    const vec3 k = vec3(-0.8660254, 0.5, 0.57735);  // hex geometry constants
    p = abs(p);
    // Reflect into fundamental hex domain (2 reflections cover all 6 sides)
    p.xy -= 2.0*min(dot(k.xy, p.xy), 0.0)*k.xy;
    vec2 d = vec2(
       length(p.xy - vec2(clamp(p.x, -k.z*h.x, k.z*h.x), h.x))*sign(p.y - h.x),
       p.z-h.y );    // d.y: distance to the top/bottom caps
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

// OCTAGON PRISM: same idea as hex prism but 8 sides.
// Two reflection planes (at 22.5° and 67.5°) fold the octagon to a wedge.
float sdOctogonPrism( in vec3 p, in float r, float h )
{
  const vec3 k = vec3(-0.9238795325,   // cos(pi/8) = sqrt(2+sqrt(2))/2
                       0.3826834323,   // sin(pi/8) = sqrt(2-sqrt(2))/2
                       0.4142135623 ); // tan(pi/8) = sqrt(2)-1
  p = abs(p);
  // Two reflections collapse 8-fold symmetry to a single wedge
  p.xy -= 2.0*min(dot(vec2( k.x,k.y),p.xy),0.0)*vec2( k.x,k.y);
  p.xy -= 2.0*min(dot(vec2(-k.x,k.y),p.xy),0.0)*vec2(-k.x,k.y);
  p.xy -= vec2(clamp(p.x, -k.z*r, k.z*r), r);  // nearest point on flat edge
  vec2 d = vec2( length(p.xy)*sign(p.y), p.z-h );
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

// CAPSULE: cylinder with hemispherical caps.
// Project p onto the segment AB: h = normalized parameter along AB.
// Clamp h to [0,1] to stay on the segment (not the infinite line).
// Distance = distance from p to the nearest point on the segment, minus radius.
float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
    vec3 pa = p-a, ba = b-a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );  // closest t on segment
    return length( pa - ba*h ) - r;
}

// ROUND CONE (axis-aligned vertical): cone whose two ends have different radii r1, r2.
// Reduce to 2D (q.x = radial distance, q.y = height).
// The lateral surface is a slanted line; b/a define its slope.
// Three regions: near bottom cap (sphere r1), near top cap (sphere r2), lateral face.
float sdRoundCone( in vec3 p, in float r1, float r2, float h )
{
    vec2 q = vec2( length(p.xz), p.y );

    float b = (r1-r2)/h;          // slope of the lateral edge in 2D
    float a = sqrt(1.0-b*b);      // cos of the cone half-angle
    float k = dot(q,vec2(-b,a));  // signed distance along the lateral normal

    if( k < 0.0 ) return length(q) - r1;     // closest to bottom sphere
    if( k > a*h ) return length(q-vec2(0.0,h)) - r2;  // closest to top sphere
    return dot(q, vec2(a,b) ) - r1;           // closest to lateral surface
}

// ROUND CONE (arbitrary orientation): same idea, generalized to 3D.
// Uses squared distances to avoid sqrt where possible (single sqrt at the end).
float sdRoundCone(vec3 p, vec3 a, vec3 b, float r1, float r2)
{
    vec3  ba = b - a;
    float l2 = dot(ba,ba);     // squared length of axis
    float rr = r1 - r2;
    float a2 = l2 - rr*rr;    // discriminant of the lateral cone geometry
    float il2 = 1.0/l2;

    vec3 pa = p - a;
    float y = dot(pa,ba);      // axial component
    float z = y - l2;
    float x2 = dot2( pa*l2 - ba*y );   // squared radial distance (scaled)
    float y2 = y*y*l2;
    float z2 = z*z*l2;

    // Classify which region p is in, then compute distance with one sqrt
    float k = sign(rr)*rr*rr*x2;
    if( sign(z)*a2*z2 > k ) return  sqrt(x2 + z2)        *il2 - r2;
    if( sign(y)*a2*y2 < k ) return  sqrt(x2 + y2)        *il2 - r1;
                            return (sqrt(x2*a2*il2)+y*rr)*il2 - r1;
}

// TRIANGULAR PRISM: 2D triangle cross-section extruded along Z.
// The triangle is handled by folding p into one sixth of the plane
// using the 3-fold symmetry of an equilateral triangle, then projecting
// onto the nearest edge.
float sdTriPrism( vec3 p, vec2 h )
{
    const float k = sqrt(3.0);
    h.x *= 0.5*k;
    p.xy /= h.x;
    p.x = abs(p.x) - 1.0;
    p.y = p.y + 1.0/k;
    if( p.x+k*p.y>0.0 ) p.xy=vec2(p.x-k*p.y,-k*p.x-p.y)/2.0;  // fold into wedge
    p.x -= clamp( p.x, -2.0, 0.0 );
    float d1 = length(p.xy)*sign(-p.y)*h.x;  // signed distance to triangle edge
    float d2 = abs(p.z)-h.y;                  // distance to top/bottom caps
    return length(max(vec2(d1,d2),0.0)) + min(max(d1,d2), 0.);
}

// CYLINDER (vertical, axis-aligned): same box trick but in cylindrical coords.
// d.x = distance to the infinite cylinder wall (radial).
// d.y = distance to the top/bottom caps (axial).
// Combine with box SDF formula for correct corners.
float sdCylinder( vec3 p, vec2 h )
{
    vec2 d = abs(vec2(length(p.xz),p.y)) - h;
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

// CYLINDER (arbitrary axis): works in the local frame of the axis AB.
// x = signed radial distance to the infinite cylinder (scaled by baba).
// y = signed axial distance to the caps (scaled by baba).
// Combine without a sqrt when inside, use Pythagoras when outside.
// The /baba at the end corrects for the scale factor.
float sdCylinder(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 pa = p - a;
    vec3 ba = b - a;
    float baba = dot(ba,ba);   // squared length of axis
    float paba = dot(pa,ba);   // projection of p onto axis (unscaled)

    float x = length(pa*baba-ba*paba) - r*baba;   // radial (scale: baba)
    float y = abs(paba-baba*0.5)-baba*0.5;         // axial  (scale: baba)
    float x2 = x*x;
    float y2 = y*y*baba;
    // Combine: avoid sqrt for interior case
    float d = (max(x,y)<0.0)?-min(x2,y2):(((x>0.0)?x2:0.0)+((y>0.0)?y2:0.0));
    return sign(d)*sqrt(abs(d))/baba;  // undo scale factor
}

// CONE (infinite, vertex at origin): reduce to 2D, project onto the slant edge.
// c = (sin, cos) of the cone half-angle. h limits the height.
// Two candidate points: projection onto the slant line (a), and onto the base cap (b).
float sdCone( in vec3 p, in vec2 c, float h )
{
    vec2 q = h*vec2(c.x,-c.y)/c.y;   // tip of the 2D slant vector
    vec2 w = vec2( length(p.xz), p.y );

    vec2 a = w - q*clamp( dot(w,q)/dot(q,q), 0.0, 1.0 );  // project onto slant
    vec2 b = w - q*vec2( clamp( w.x/q.x, 0.0, 1.0 ), 1.0 );  // project onto base
    float k = sign( q.y );
    float d = min(dot( a, a ),dot(b, b));
    float s = max( k*(w.x*q.y-w.y*q.x),k*(w.y-q.y)  );
    return sqrt(d)*sign(s);
}

// CAPPED CONE (axis-aligned): cone frustum with flat top and bottom.
// ca = closest point vector to the top/bottom caps.
// cb = closest point vector to the slant surface (clamped parametric projection).
// Sign is negative inside (both ca.y<0 AND cb.x<0).
float sdCappedCone( in vec3 p, in float h, in float r1, in float r2 )
{
    vec2 q = vec2( length(p.xz), p.y );

    vec2 k1 = vec2(r2,h);
    vec2 k2 = vec2(r2-r1,2.0*h);
    vec2 ca = vec2(q.x-min(q.x,(q.y < 0.0)?r1:r2), abs(q.y)-h);
    vec2 cb = q - k1 + k2*clamp( dot(k1-q,k2)/dot2(k2), 0.0, 1.0 );
    float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return s*sqrt( min(dot2(ca),dot2(cb)) );
}

// CAPPED CONE (arbitrary axis): same idea generalized to 3D.
// Parameterize along the axis (paba ∈ [0,1]).
// x = radial distance from the axis at height paba.
// ca/cb = two candidate closest point vectors; take the minimum.
float sdCappedCone(vec3 p, vec3 a, vec3 b, float ra, float rb)
{
    float rba  = rb-ra;
    float baba = dot(b-a,b-a);
    float papa = dot(p-a,p-a);
    float paba = dot(p-a,b-a)/baba;   // normalized axial parameter [0,1]

    float x = sqrt( papa - paba*paba*baba );   // radial distance from axis

    float cax = max(0.0,x-((paba<0.5)?ra:rb)); // excess beyond the cap radius
    float cay = abs(paba-0.5)-0.5;             // axial: negative = inside cap slab

    float k = rba*rba + baba;
    float f = clamp( (rba*(x-ra)+paba*baba)/k, 0.0, 1.0 );  // closest t on slant

    float cbx = x-ra - f*rba;
    float cby = paba - f;

    float s = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;

    return s*sqrt( min(cax*cax + cay*cay*baba,
                       cbx*cbx + cby*cby*baba) );
}

// SOLID ANGLE: the intersection of a sphere and an infinite cone.
// Looks like a bite taken out of a sphere.
// l = distance to sphere surface (negative inside sphere).
// m = distance to the cone's lateral surface.
// max(l, m*sign(...)) combines them: point must be inside both sphere AND cone.
float sdSolidAngle(vec3 pos, vec2 c, float ra)
{
    vec2 p = vec2( length(pos.xz), pos.y );
    float l = length(p) - ra;     // distance to sphere
    float m = length(p - c*clamp(dot(p,c),0.0,ra) );  // distance to cone edge arc
    return max(l,m*sign(c.y*p.x-c.x*p.y));  // intersection of sphere and cone
}

// OCTAHEDRON: |x|+|y|+|z|=s is a plane in the folded octant.
// Fold p into the positive octant, then classify which face is closest
// by checking which axis has the smallest contribution to the L1 norm.
// 0.57735027 = 1/sqrt(3) = distance scale for the face normal.
float sdOctahedron(vec3 p, float s)
{
    p = abs(p);    // fold into positive octant (8-fold symmetry)
    float m = p.x + p.y + p.z - s;   // L1 distance proxy

    // Find which face p projects onto (the face with the smallest component)
    vec3 q;
         if( 3.0*p.x < m ) q = p.xyz;
    else if( 3.0*p.y < m ) q = p.yzx;
    else if( 3.0*p.z < m ) q = p.zxy;
    else return m*0.57735027;    // corner region: L1 distance scaled to Euclidean
    float k = clamp(0.5*(q.z-q.y+s),0.0,s);
    return length(vec3(q.x,q.y-s+k,q.z-k));
}

// PYRAMID (square base): exploit 4-fold symmetry, then project onto a triangular face.
// m2 = h^2 + 0.25 comes from the face normal of a unit-base pyramid.
// q is a rotated coordinate frame aligned with one triangular face.
// Two candidate distances (a, b) cover the two edges of the triangular face.
float sdPyramid( in vec3 p, in float h )
{
    float m2 = h*h + 0.25;   // squared face normal magnitude

    // Fold XZ into the first quadrant and sort so p.x >= p.z (symmetry)
    p.xz = abs(p.xz);
    p.xz = (p.z>p.x) ? p.zx : p.xz;
    p.xz -= 0.5;   // shift to corner of base triangle

    // Rotate into face-aligned 2D coordinate system
    vec3 q = vec3( p.z, h*p.y - 0.5*p.x, h*p.x + 0.5*p.y);

    float s = max(-q.x,0.0);
    float t = clamp( (q.y-0.5*p.z)/(m2+0.25), 0.0, 1.0 );

    float a = m2*(q.x+s)*(q.x+s) + q.y*q.y;          // dist^2 to base edge
    float b = m2*(q.x+0.5*t)*(q.x+0.5*t) + (q.y-m2*t)*(q.y-m2*t);  // dist^2 to apex edge

    float d2 = min(q.y,-q.x*m2-q.y*0.5) > 0.0 ? 0.0 : min(a,b);

    return sqrt( (d2+q.z*q.z)/m2 ) * sign(max(q.z,-p.y));
}

// RHOMBUS: a rounded diamond shape (rhombus cross-section extruded vertically).
// ndot projects onto the diagonal axes of the rhombus (not the standard axes).
// f = interpolation parameter along the rhombus perimeter.
// Then reduce to a 2D box SDF in (radial offset from perimeter, height).
float sdRhombus(vec3 p, float la, float lb, float h, float ra)
{
    p = abs(p);
    vec2 b = vec2(la,lb);
    float f = clamp( (ndot(b,b-2.0*p.xz))/dot(b,b), -1.0, 1.0 );  // perimeter param
    vec2 q = vec2(length(p.xz-0.5*b*vec2(1.0-f,1.0+f))*sign(p.x*b.y+p.z*b.x-b.x*b.y)-ra, p.y-h);
    return min(max(q.x,q.y),0.0) + length(max(q,0.0));
}

// HORSESHOE: a U-shaped solid (open torus with straight legs).
// 1. Fold x-axis symmetry.
// 2. Rotate the arc portion into a canonical frame using the opening angle.
// 3. The straight leg region uses the untransformed radial length l.
// 4. Offset by arc radius r and leg length le, then measure distance to
//    a rectangular cross-section w.
float sdHorseshoe( in vec3 p, in vec2 c, in float r, in float le, vec2 w )
{
    p.x = abs(p.x);   // left-right symmetry
    float l = length(p.xy);
    // Rotate by the opening angle (c = cos/sin pair)
    p.xy = mat2(-c.x, c.y,
              c.y, c.x)*p.xy;
    // Select arc vs. leg region: use rotated coords inside arc, radial length outside
    p.xy = vec2((p.y>0.0 || p.x>0.0)?p.x:l*sign(-c.x),
                (p.x>0.0)?p.y:l );
    // Offset to the cross-section center
    p.xy = vec2(p.x,abs(p.y-r))-vec2(le,0.0);

    // Distance to rectangular cross-section (box SDF in 2D)
    vec2 q = vec2(length(max(p.xy,0.0)) + min(0.0,max(p.x,p.y)),p.z);
    vec2 d = abs(q) - w;
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

// U-SHAPE: like a horseshoe but with sharp 180° turn (semicircle bottom).
// The top portion is two parallel vertical legs (abs(x)), the bottom
// is a semicircle (length(xy)). The condition p.y>0 selects between them.
float sdU( in vec3 p, in float r, in float le, vec2 w )
{
    p.x = (p.y>0.0) ? abs(p.x) : length(p.xy);  // leg vs. arc
    p.x = abs(p.x-r);   // radial distance from the tube spine
    p.y = p.y - le;     // shift up by leg length
    float k = max(p.x,p.y);
    vec2 q = vec2( (k<0.0) ? -k : length(max(p.xy,0.0)), abs(p.z) ) - w;
    return length(max(q,0.0)) + min(max(q.x,q.y),0.0);
}

// =============================================================================
// SCENE COMPOSITION
// =============================================================================

// opU: UNION of two SDFs. Returns the closer surface.
// The vec2 carries (distance, material_id) together.
// This is the key Boolean CSG operation: min(d1,d2) takes the nearer surface.
vec2 opU( vec2 d1, vec2 d2 )
{
    return (d1.x<d2.x) ? d1 : d2;
}

// ZERO TRICK: prevents the GLSL compiler from unrolling loops.
// The compiler cannot prove min(iFrame,0)==0 at compile time (iFrame is a uniform),
// so it cannot statically unroll loops that start at ZERO.
// Unrolling would massively bloat the shader binary and hurt GPU register pressure.
#define ZERO (min(iFrame,0))

// map(): the complete scene SDF.
// Returns vec2(distance, material_id).
// material_id is used later to assign color/material properties.
//
// BOUNDING BOX OPTIMIZATION:
// Each group of objects is wrapped in a cheap sdBox test.
// If the marching point is outside the bounding box (sdBox > current best distance),
// we skip all the SDFs inside it entirely.
// This is valid because the SDF is conservative: if we're farther from the box
// than our current best hit, the contents cannot be closer.
// This is O(1) rejection per group, equivalent to a very simple BVH.
vec2 map( in vec3 pos )
{
    // Start with the ground plane (material 0.0 = floor)
    vec2 res = vec2( pos.y, 0.0 );

    // Group 1: sphere and rhombus
    if( sdBox( pos-vec3(-2.0,0.3,0.25),vec3(0.3,0.3,1.0) )<res.x )
    {
      res = opU( res, vec2( sdSphere(    pos-vec3(-2.0,0.25, 0.0), 0.25 ), 26.9 ) );
      res = opU( res, vec2( sdRhombus(  (pos-vec3(-2.0,0.25, 1.0)).xzy, 0.15, 0.25, 0.04, 0.08 ),17.0 ) );
    }

    // Group 2: torus variants, box frame, cone, capped cone, solid angle
    if( sdBox( pos-vec3(0.0,0.3,-1.0),vec3(0.35,0.3,2.5) )<res.x )
    {
      res = opU( res, vec2( sdCappedTorus((pos-vec3( 0.0,0.30, 1.0))*vec3(1,-1,1), vec2(0.866025,-0.5), 0.25, 0.05), 25.0) );
      res = opU( res, vec2( sdBoxFrame(    pos-vec3( 0.0,0.25, 0.0), vec3(0.3,0.25,0.2), 0.025 ), 16.9 ) );
      res = opU( res, vec2( sdCone(        pos-vec3( 0.0,0.45,-1.0), vec2(0.6,0.8),0.45 ), 55.0 ) );
      res = opU( res, vec2( sdCappedCone(  pos-vec3( 0.0,0.25,-2.0), 0.25, 0.25, 0.1 ), 13.67 ) );
      res = opU( res, vec2( sdSolidAngle(  pos-vec3( 0.0,0.00,-3.0), vec2(3,4)/5.0, 0.4 ), 49.13 ) );
    }

    // Group 3: torus, box, capsule, cylinder, hex prism
    if( sdBox( pos-vec3(1.0,0.3,-1.0),vec3(0.35,0.3,2.5) )<res.x )
    {
      res = opU( res, vec2( sdTorus(      (pos-vec3( 1.0,0.30, 1.0)).xzy, vec2(0.25,0.05) ), 7.1 ) );
      res = opU( res, vec2( sdBox(         pos-vec3( 1.0,0.25, 0.0), vec3(0.3,0.25,0.1) ), 3.0 ) );
      res = opU( res, vec2( sdCapsule(     pos-vec3( 1.0,0.00,-1.0),vec3(-0.1,0.1,-0.1), vec3(0.2,0.4,0.2), 0.1  ), 31.9 ) );
      res = opU( res, vec2( sdCylinder(    pos-vec3( 1.0,0.25,-2.0), vec2(0.15,0.25) ), 8.0 ) );
      res = opU( res, vec2( sdHexPrism(    pos-vec3( 1.0,0.2,-3.0), vec2(0.2,0.05) ), 18.4 ) );
    }

    // Group 4: pyramid, octahedron, tri prism, ellipsoid, horseshoe
    if( sdBox( pos-vec3(-1.0,0.35,-1.0),vec3(0.35,0.35,2.5))<res.x )
    {
      res = opU( res, vec2( sdPyramid(    pos-vec3(-1.0,-0.6,-3.0), 1.0 ), 13.56 ) );
      res = opU( res, vec2( sdOctahedron( pos-vec3(-1.0,0.15,-2.0), 0.35 ), 23.56 ) );
      res = opU( res, vec2( sdTriPrism(   pos-vec3(-1.0,0.15,-1.0), vec2(0.3,0.05) ),43.5 ) );
      res = opU( res, vec2( sdEllipsoid(  pos-vec3(-1.0,0.25, 0.0), vec3(0.2, 0.25, 0.05) ), 43.17 ) );
      res = opU( res, vec2( sdHorseshoe(  pos-vec3(-1.0,0.25, 1.0), vec2(cos(1.3),sin(1.3)), 0.2, 0.3, vec2(0.03,0.08) ), 11.5 ) );
    }

    // Group 5: octagon prism, arbitrary-axis cylinder, capped cone, round cones
    if( sdBox( pos-vec3(2.0,0.3,-1.0),vec3(0.35,0.3,2.5) )<res.x )
    {
      res = opU( res, vec2( sdOctogonPrism(pos-vec3( 2.0,0.2,-3.0), 0.2, 0.05), 51.8 ) );
      res = opU( res, vec2( sdCylinder(    pos-vec3( 2.0,0.14,-2.0), vec3(0.1,-0.1,0.0), vec3(-0.2,0.35,0.1), 0.08), 31.2 ) );
      res = opU( res, vec2( sdCappedCone(  pos-vec3( 2.0,0.09,-1.0), vec3(0.1,0.0,0.0), vec3(-0.2,0.40,0.1), 0.15, 0.05), 46.1 ) );
      res = opU( res, vec2( sdRoundCone(   pos-vec3( 2.0,0.15, 0.0), vec3(0.1,0.0,0.0), vec3(-0.1,0.35,0.1), 0.15, 0.05), 51.7 ) );
      res = opU( res, vec2( sdRoundCone(   pos-vec3( 2.0,0.20, 1.0), 0.2, 0.1, 0.3 ), 37.0 ) );
    }

    return res;
}

// =============================================================================
// RAY-AABB INTERSECTION
// =============================================================================
// iBox: analytically intersects a ray with an axis-aligned box.
// Returns vec2(t_enter, t_exit). If t_enter > t_exit the ray misses the box.
// Used as a broad-phase cull: only sphere-trace inside the bounding box
// of all primitives, avoiding wasted march steps in empty space outside.
//
// Derivation: for each axis, the ray crosses the two slab planes at:
//   t = (±half_size - ray_origin) / ray_dir  =  -n ± k
// where m=1/rd (reciprocal direction), n=m*ro (origin in slab space), k=|m|*rad.
// t_enter = max of the three entry times, t_exit = min of the three exit times.
vec2 iBox( in vec3 ro, in vec3 rd, in vec3 rad )
{
    vec3 m = 1.0/rd;        // reciprocal direction (precomputed for all 3 axes)
    vec3 n = m*ro;          // origin projected into slab space
    vec3 k = abs(m)*rad;    // half-extent in slab space
    vec3 t1 = -n - k;       // entry times per axis
    vec3 t2 = -n + k;       // exit times per axis
    return vec2( max( max( t1.x, t1.y ), t1.z ),   // latest entry = actual entry
                 min( min( t2.x, t2.y ), t2.z ) );  // earliest exit = actual exit
}

// =============================================================================
// SPHERE TRACING (Raymarching)
// =============================================================================
// The core algorithm. For a ray ro + t*rd:
//   1. Query the scene SDF at the current point: h = map(p).
//   2. The SDF guarantees no surface is closer than h units along any direction.
//      So we can safely advance t by h without missing any surface.
//   3. Repeat until h < epsilon (surface hit) or t > tmax (miss).
//
// Convergence: near surfaces, h → 0 and steps become tiny, but the
// termination condition abs(h) < 0.0001*t (relative epsilon) handles this.
// The relative epsilon 0.0001*t allows more tolerance at distance (where
// one pixel covers more world space), preventing "false hits" at grazing angles.
//
// Maximum 70 steps is enough for this scene. Complex/thin SDFs may need more.
vec2 raycast( in vec3 ro, in vec3 rd )
{
    vec2 res = vec2(-1.0,-1.0);   // default: no hit

    float tmin = 1.0;    // start away from camera (avoid self-intersection)
    float tmax = 20.0;   // maximum scene depth

    // First: check the ground plane analytically (ray-plane intersection).
    // A ray hits y=0 when ro.y + t*rd.y = 0 → t = -ro.y/rd.y.
    // If tp1 > 0 the ray travels toward the ground; tighten tmax so we don't
    // march past it (the plane SDF is globally valid, no need to sphere-trace it).
    float tp1 = (0.0-ro.y)/rd.y;
    if( tp1>0.0 )
    {
        tmax = min( tmax, tp1 );
        res = vec2( tp1, 1.0 );   // tentative floor hit
    }

    // Broad-phase: intersect the ray with the bounding box of all primitives.
    // Only march inside the box interval [tb.x, tb.y].
    // This skips all marching in empty space outside the scene cluster.
    vec2 tb = iBox( ro-vec3(0.0,0.4,-0.5), rd, vec3(2.5,0.41,3.0) );
    if( tb.x<tb.y && tb.y>0.0 && tb.x<tmax)
    {
        tmin = max(tb.x,tmin);
        tmax = min(tb.y,tmax);

        float t = tmin;
        for( int i=0; i<70 && t<tmax; i++ )
        {
            vec2 h = map( ro+rd*t );
            // Relative epsilon: at distance t, we accept a hit if within 0.01% of t.
            // This prevents infinite marching near grazing incidence on flat surfaces.
            if( abs(h.x)<(0.0001*t) )
            {
                res = vec2(t,h.y);
                break;
            }
            t += h.x;   // safe step: h.x is the minimum distance to any surface
        }
    }

    return res;
}

// =============================================================================
// SOFT SHADOW APPROXIMATION
// =============================================================================
// TRICK: instead of sampling the area of the light source (which would require
// many rays = Monte Carlo), march a single ray toward the light and track how
// close it gets to any occluder.
//
// At each step t along the shadow ray, h = map(p) is the distance to the
// nearest surface. The ratio h/t approximates the sine of the angular gap
// between the shadow ray and the nearest occluder.
// Scaled by 8.0 and clamped, this becomes a soft penumbra factor in [0,1]:
//   s = 1.0  → fully lit (no nearby occluders)
//   s ≈ 0.0  → deep shadow
//
// The result is smoothstepped with res*res*(3-2*res) to soften the penumbra edge.
// This is NOT physically correct for area lights, but it is O(24 steps) instead
// of O(N²) Monte Carlo samples, and it looks convincing.
float calcSoftshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
    // Tighten tmax at the sky plane so we don't waste steps above y=0.8
    float tp = (0.8-ro.y)/rd.y; if( tp>0.0 ) tmax = min( tmax, tp );

    float res = 1.0;   // start fully lit
    float t = mint;
    for( int i=ZERO; i<24; i++ )
    {
        float h = map( ro + rd*t ).x;
        // h/t ≈ sin(angular gap to occluder). Scale by 8 for penumbra width.
        float s = clamp(8.0*h/t,0.0,1.0);
        res = min( res, s );     // track the darkest point along the ray
        t += clamp( h, 0.01, 0.2 );  // step: sphere trace but clamp min step to avoid crawl
        if( res<0.004 || t>tmax ) break;  // early exit: fully shadowed or past light
    }
    res = clamp( res, 0.0, 1.0 );
    return res*res*(3.0-2.0*res);   // smoothstep to soften the penumbra transition
}

// =============================================================================
// NORMAL ESTIMATION (Tetrahedron Gradient)
// =============================================================================
// TRICK: the gradient of an SDF equals the surface normal.
// Approximate it with finite differences: normal ≈ ∇map(pos).
//
// Naive approach uses 6 map() calls (±ε per axis).
// IQ's trick uses only 4 calls with a tetrahedron stencil:
//   e0 = (+,+,+),  e1 = (-,-,+),  e2 = (-,+,-),  e3 = (+,-,-)
// These 4 directions are the vertices of a regular tetrahedron inscribed in a cube.
// The weighted sum exactly recovers the gradient (cross-terms cancel).
// 0.5773 ≈ 1/√3 normalizes the tetrahedron vertices to unit length.
// ε = 0.0005 is small enough to be accurate but large enough to avoid float precision issues.
vec3 calcNormal( in vec3 pos )
{
    vec3 n = vec3(0.0);
    for( int i=ZERO; i<4; i++ )
    {
        // Generate the 4 tetrahedron vertices from bit patterns of i:
        //   i=0: bits 11,10,00 → (+1,+1,-1) after (2*bit-1) → (+,+,-)... wait:
        //   The bit encoding: (((i+3)>>1)&1), ((i>>1)&1), (i&1) for i=0,1,2,3
        //   gives: (1,0,0),(1,1,1),(0,1,0),(0,0,1) → after *2-1: (+,-,-),(-,+,-),(-,-,+),(+,+,+)?
        //   Actually gives the 4 tetrahedron vertices correctly (they satisfy x*y*z=+1 alternating).
        vec3 e = 0.5773*(2.0*vec3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        n += e*map(pos+0.0005*e).x;   // weighted by SDF value at offset position
    }
    return normalize(n);
}

// =============================================================================
// AMBIENT OCCLUSION APPROXIMATION
// =============================================================================
// TRICK: instead of integrating visibility over the hemisphere (which requires
// Monte Carlo with many rays), march 5 samples along the surface normal.
//
// At distance h along the normal, the SDF value d should equal h if space is open.
// If d < h, the surface curves back and occludes: the shortfall (h-d) accumulates.
// Exponential weight sca *= 0.95 gives more importance to closer occluders.
//
// This is NOT the correct AO integral (∫ V(ω)(ω·n) dω over hemisphere),
// but it is a plausible and cheap approximation: O(5) map() calls vs. O(100+).
//
// The final (0.5+0.5*nor.y) term adds a directional bias: surfaces facing up
// get slightly less AO reduction (they're more exposed to the sky).
float calcAO( in vec3 pos, in vec3 nor )
{
    float occ = 0.0;
    float sca = 1.0;   // exponentially decreasing weight
    for( int i=ZERO; i<5; i++ )
    {
        float h = 0.01 + 0.12*float(i)/4.0;   // step distances: 0.01 to 0.13
        float d = map( pos + h*nor ).x;
        occ += (h-d)*sca;    // accumulate "blocked" fraction
        sca *= 0.95;         // further samples contribute less
        if( occ>0.35 ) break;  // early exit: already heavily occluded
    }
    return clamp( 1.0 - 3.0*occ, 0.0, 1.0 ) * (0.5+0.5*nor.y);
}

// =============================================================================
// ANTI-ALIASED CHECKERBOARD
// =============================================================================
// TRICK: instead of sampling the checkerboard (which would alias at a distance),
// analytically integrate it over the pixel footprint.
//
// dpdx/dpdy are the derivatives of the world-space position with respect to
// screen-space x/y (the pixel footprint projected onto the floor plane).
// w = footprint size; i = analytical box-filter integral of the sign() pattern.
// The result is a pre-filtered checkerboard with no Moire patterns or shimmer.
// This is how texture anti-aliasing SHOULD work (equivalent to hardware mipmapping
// but computed analytically for an infinite procedural pattern).
float checkersGradBox( in vec2 p, in vec2 dpdx, in vec2 dpdy )
{
    vec2 w = abs(dpdx)+abs(dpdy) + 0.001;   // footprint width (L1 size of pixel)
    // Analytical integral of the |fract(p)-0.5| pattern over [p-w/2, p+w/2]
    vec2 i = 2.0*(abs(fract((p-0.5*w)*0.5)-0.5)-abs(fract((p+0.5*w)*0.5)-0.5))/w;
    return 0.5 - 0.5*i.x*i.y;   // XOR pattern: 0 or 1 smoothly blended at edges
}

// =============================================================================
// SHADING / RENDER
// =============================================================================
// This function replaces the rendering equation integral with 4 analytical terms.
// NONE of them solve ∫ fr(ωi,ωo) Li(ωi) cosθ dωi — they are all heuristics
// designed to look physically plausible while being O(1) or O(small constant).
vec3 render( in vec3 ro, in vec3 rd, in vec3 rdx, in vec3 rdy )
{
    // BACKGROUND: simple sky gradient. Bluer at the horizon (rd.y=0), slightly
    // darker overhead (rd.y=1). No actual sky model — just a linear formula.
    vec3 col = vec3(0.7, 0.7, 0.9) - max(rd.y,0.0)*0.3;

    vec2 res = raycast(ro,rd);
    float t = res.x;
    float m = res.y;   // material ID
    if( m>-0.5 )       // something was hit (m=-1 means no hit)
    {
        vec3 pos = ro + t*rd;   // hit position

        // NORMAL: floor is always (0,1,0); other surfaces use the tetrahedron trick.
        // The floor normal is exact and free (no map() calls needed).
        vec3 nor = (m<1.5) ? vec3(0.0,1.0,0.0) : calcNormal( pos );
        vec3 ref = reflect( rd, nor );   // perfect mirror reflection direction

        // MATERIAL COLOR: encode hue from material ID using sin(m*2 + phase).
        // Each shape gets a unique color by having a unique fractional material ID.
        // No texture lookups — just a trigonometric hash of the material ID.
        col = 0.2 + 0.2*sin( m*2.0 + vec3(0.0,1.0,2.0) );
        float ks = 1.0;   // specular strength

        if( m<1.5 )   // floor plane
        {
            // FLOOR ANTI-ALIASED CHECKERBOARD:
            // Project the pixel footprint (rdx-rd and rdy-rd) onto the floor plane.
            // dpdx/dpdy = how much world position changes per screen pixel (on y=0 plane).
            // This is used by checkersGradBox for proper box filtering.
            vec3 dpdx = ro.y*(rd/rd.y-rdx/rdx.y);
            vec3 dpdy = ro.y*(rd/rd.y-rdy/rdy.y);

            float f = checkersGradBox( 3.0*pos.xz, 3.0*dpdx.xz, 3.0*dpdy.xz );
            col = 0.15 + f*vec3(0.05);   // dark/light checker squares
            ks = 0.4;   // floor is less specular than objects
        }

        // AO: computed once, reused by all lighting terms.
        // Modulates each light contribution to fake indirect shadowing.
        float occ = calcAO( pos, nor );

        vec3 lin = vec3(0.0);   // accumulated irradiance

        // ------------------------------------------------------------------
        // LIGHT TERM 1: SUN (directional light)
        // Replaces: ∫ fr(ωi,ωo) Li_sun(ωi) cosθ dωi  (over a delta direction)
        // A delta (point at infinity) light has no area, so no integral needed.
        // ------------------------------------------------------------------
        {
            vec3  lig = normalize( vec3(-0.5, 0.4, -0.6) );  // sun direction (fixed)

            // Blinn-Phong halfway vector: bisector between light and view directions.
            // Used for specular highlight: cheaper than computing the reflection.
            vec3  hal = normalize( lig-rd );

            // DIFFUSE: Lambert cosine law. dot(nor, lig) = cosine of angle of incidence.
            // Clamped to [0,1]: backlit faces get 0 (no negative light).
            float dif = clamp( dot( nor, lig ), 0.0, 1.0 );

            // Multiply by soft shadow: approximates partial occlusion from the sun.
            // A physically correct area-light shadow would need many shadow rays.
            dif *= calcSoftshadow( pos, lig, 0.02, 2.5 );

            // SPECULAR (Blinn-Phong): pow(dot(nor,hal), 16) = specular lobe.
            // Exponent 16 = moderately shiny. Higher = sharper highlight.
            float spe = pow( clamp( dot( nor, hal ), 0.0, 1.0 ),16.0);
            // Specular only where there's diffuse light (physically motivated)
            spe *= dif;
            // FRESNEL (Schlick approximation):
            //   F(θ) ≈ F0 + (1-F0)*(1-cosθ)^5
            // F0=0.04 is the reflectance at normal incidence for typical dielectrics (glass, plastic).
            // At grazing angles (dot(hal,lig)→0), Fresnel → 1.0: more specular.
            // At normal incidence (dot(hal,lig)→1), Fresnel → 0.04: mostly diffuse.
            spe *= 0.04+0.96*pow(clamp(1.0-dot(hal,lig),0.0,1.0),5.0);

            // Accumulate: warm yellow-white sun color (1.3, 1.0, 0.7).
            lin += col*2.20*dif*vec3(1.30,1.00,0.70);   // diffuse contribution
            lin +=     5.00*spe*vec3(1.30,1.00,0.70)*ks; // specular contribution
        }

        // ------------------------------------------------------------------
        // LIGHT TERM 2: SKY (ambient / indirect)
        // Replaces: ∫_hemisphere fr(ωi,ωo) Li_sky(ωi) cosθ dωi
        // A proper sky integral needs Monte Carlo (hemisphere sampling).
        // APPROXIMATION: assume uniform blue sky radiance, then the Lambertian
        // integral over the upper hemisphere gives a result proportional to
        // (0.5 + 0.5*nor.y). The sqrt() makes the falloff look softer (sky light
        // is brightest overhead, not exactly Lambertian in appearance).
        // Multiplied by AO: if geometry blocks the sky, reduce sky contribution.
        // ------------------------------------------------------------------
        {
            // sqrt(0.5+0.5*nor.y): approximates ∫_upper_hemisphere cosθ dω
            // for a uniform sky. Range: 0 (facing down) to 1 (facing up).
            float dif = sqrt(clamp( 0.5+0.5*nor.y, 0.0, 1.0 ));
            dif *= occ;   // AO modulates sky visibility

            // SKY SPECULAR: check if the reflection direction points into the sky.
            // smoothstep(-0.2, 0.2, ref.y): soft threshold — reflections that point
            // slightly below horizon still get some sky color (approximates the real
            // sky-to-ground color gradient).
            float spe = smoothstep( -0.2, 0.2, ref.y );
            spe *= dif;
            // Fresnel for sky specular: same Schlick formula but using view/normal angle
            spe *= 0.04+0.96*pow(clamp(1.0+dot(nor,rd),0.0,1.0), 5.0 );
            // Shadow the reflection ray against the scene (e.g., if reflecting into an object)
            spe *= calcSoftshadow( pos, ref, 0.02, 2.5 );

            lin += col*0.60*dif*vec3(0.40,0.60,1.15);   // blue sky diffuse
            lin +=     2.00*spe*vec3(0.40,0.60,1.30)*ks; // sky specular
        }

        // ------------------------------------------------------------------
        // LIGHT TERM 3: BACK LIGHT (fill light from behind/below)
        // A second fake directional light from the opposite side of the sun.
        // Simulates light bounced off the ground or a back wall.
        // Physically there is no such light — it replaces global illumination.
        // Clamped by (1-pos.y): only low objects get fill light (fades out
        // as objects rise above the ground, where the bounce would be weaker).
        // ------------------------------------------------------------------
        {
            float dif = clamp( dot( nor, normalize(vec3(0.5,0.0,0.6))), 0.0, 1.0 )
                       *clamp( 1.0-pos.y,0.0,1.0);   // attenuate with height
            dif *= occ;
            lin += col*0.55*dif*vec3(0.25,0.25,0.25);   // dim grey fill light
        }

        // ------------------------------------------------------------------
        // LIGHT TERM 4: SSS / RIM LIGHT (subsurface scattering approximation)
        // pow(1 + dot(nor,rd), 2): maximum when nor points toward viewer (backlit rim).
        // This fakes the "glow" of translucent materials (wax, skin, leaves) where
        // light scatters through and exits toward the viewer.
        // Not physically based — it's an artistic rim light effect.
        // ------------------------------------------------------------------
        {
            // 1 + dot(nor,rd): dot(nor,rd) ≈ -1 when ray hits front face (=0),
            // ≈ 0 at grazing angle (=1). So this is strongest at grazing/backlit angles.
            float dif = pow(clamp(1.0+dot(nor,rd),0.0,1.0),2.0);
            dif *= occ;
            lin += col*0.25*dif*vec3(1.00,1.00,1.00);   // white rim/SSS glow
        }

        col = lin;

        // ATMOSPHERIC FOG: exponential depth fog.
        // mix(col, fogColor, 1 - exp(-k*t^3)): cubic distance fog.
        // t^3 makes fog onset very gradual (nearly no fog close up) then
        // rapid (heavy fog far away). The sky color (0.7,0.7,0.9) matches
        // the background so objects smoothly fade into it at distance.
        col = mix( col, vec3(0.7,0.7,0.9), 1.0-exp( -0.0001*t*t*t ) );
    }

    return vec3( clamp(col,0.0,1.0) );
}

// =============================================================================
// CAMERA
// =============================================================================
// Constructs an orthonormal camera basis (right, up, forward) from
// a look-at target. cr = camera roll angle.
// 1. cw = forward vector (normalized from eye to target)
// 2. cp = "up reference" (slightly tilted by roll angle cr)
// 3. cu = right = cross(forward, up_ref), normalized
// 4. cv = up = cross(right, forward)  (no normalize needed: already unit since cu⊥cw)
// Returns a matrix whose columns are (right, up, forward).
// Multiply by a local ray direction to get the world-space ray.
mat3 setCamera( in vec3 ro, in vec3 ta, float cr )
{
    vec3 cw = normalize(ta-ro);                         // forward
    vec3 cp = vec3(sin(cr), cos(cr),0.0);               // up reference (with roll)
    vec3 cu = normalize( cross(cw,cp) );                // right
    vec3 cv =          ( cross(cu,cw) );                // up (already normalized)
    return mat3( cu, cv, cw );
}

// =============================================================================
// MAIN IMAGE
// =============================================================================
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Mouse controls camera azimuth (horizontal orbit).
    // mo.x in [0,1] maps to a full orbit via the 7.0*mo.x in the camera position.
    vec2 mo = iMouse.xy/iResolution.xy;

    // Start at time=32 so the camera begins at a nice angle (not t=0 facing a bad direction).
    float time = 32.0 + iTime*1.5;

    // CAMERA ORBIT: circular path around the look-at target.
    // ro orbits at radius 4.5, height 2.2. The 0.1*time makes it slow.
    vec3 ta = vec3( 0.25, -0.75, -0.75 );   // look-at target
    vec3 ro = ta + vec3( 4.5*cos(0.1*time + 7.0*mo.x), 2.2, 4.5*sin(0.1*time + 7.0*mo.x) );
    mat3 ca = setCamera( ro, ta, 0.0 );

    vec3 tot = vec3(0.0);   // accumulated color (for AA)

    // GRID SUPERSAMPLING: AA×AA evenly spaced subpixel samples.
    // Unlike Monte Carlo AA (random samples), grid supersampling is deterministic
    // and has no noise — each pixel always converges to the same value.
    // Drawback: only 4 fixed sample positions (2x2), cannot capture high-frequency
    // patterns that alias between those positions. But for this scene it's enough.
#if AA>1
    for( int m=ZERO; m<AA; m++ )
    for( int n=ZERO; n<AA; n++ )
    {
        // o offsets the fragCoord by a fraction of a pixel in each direction.
        // Range: [-0.5, +0.5) pixels in each axis.
        vec2 o = vec2(float(m),float(n)) / float(AA) - 0.5;
        // Map pixel to NDC: x ∈ [-aspect, +aspect], y ∈ [-1, +1]
        // Dividing by iResolution.y (not .x) keeps square pixels regardless of aspect ratio.
        vec2 p = (2.0*(fragCoord+o)-iResolution.xy)/iResolution.y;
#else
        vec2 p = (2.0*fragCoord-iResolution.xy)/iResolution.y;
#endif

        // FOCAL LENGTH: fl=2.5 gives a ~44° field of view (tan(fov/2) = 1/fl).
        // Larger fl = narrower FOV (telephoto). This is effectively perspective projection.
        const float fl = 2.5;

        // Primary ray direction: transform NDC (p.x, p.y, fl) by camera matrix.
        // normalize() ensures a unit direction vector.
        vec3 rd = ca * normalize( vec3(p,fl) );

        // RAY DIFFERENTIALS: rays to the neighboring pixels (x+1, y+1).
        // rdx and rdy define the pixel footprint in world space.
        // Used in checkersGradBox to compute the area the pixel covers on the floor,
        // enabling analytic anti-aliasing of the checkerboard pattern.
        // This is cheaper than actually sampling the floor multiple times per pixel.
        vec2 px = (2.0*(fragCoord+vec2(1.0,0.0))-iResolution.xy)/iResolution.y;
        vec2 py = (2.0*(fragCoord+vec2(0.0,1.0))-iResolution.xy)/iResolution.y;
        vec3 rdx = ca * normalize( vec3(px,fl) );
        vec3 rdy = ca * normalize( vec3(py,fl) );

        vec3 col = render( ro, rd, rdx, rdy );

        // GAMMA CORRECTION: monitor displays in sRGB (approximately gamma 2.2).
        // Lighting is computed in linear light space. To display correctly,
        // convert linear → sRGB by raising to power 1/2.2 ≈ 0.4545.
        // Without this: image looks washed out / too bright in midtones.
        col = pow( col, vec3(0.4545) );

        tot += col;
#if AA>1
    }
    // Average all AA^2 samples to get the final pixel color.
    tot /= float(AA*AA);
#endif

    fragColor = vec4( tot, 1.0 );
}

void main()
{
    mainImage(outColor, gl_FragCoord.xy);
}
)GLSL";

// -----------------------------------------------------------------------------

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

static GLuint compile_shader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        char log[4096];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        fprintf(stderr, "Shader compile error:\n%s\n", log);
    }
    return s;
}

static GLuint create_program(const char* vert_src, const char* frag_src)
{
    GLuint vert = compile_shader(GL_VERTEX_SHADER,   vert_src);
    GLuint frag = compile_shader(GL_FRAGMENT_SHADER, frag_src);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        char log[4096];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        fprintf(stderr, "Program link error:\n%s\n", log);
    }
    glDeleteShader(vert);
    glDeleteShader(frag);
    return prog;
}

// Main code
int main(int, char**)
{
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100 (WebGL 1.0)
    const char* glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(IMGUI_IMPL_OPENGL_ES3)
    // GL ES 3.0 + GLSL 300 es (WebGL 2.0)
    const char* glsl_version = "#version 300 es";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

    // Create window with graphics context
    float main_scale = ImGui_ImplGlfw_GetContentScaleForMonitor(glfwGetPrimaryMonitor()); // Valid on GLFW 3.3+ only
    GLFWwindow* window = glfwCreateWindow((int)(1280 * main_scale), (int)(800 * main_scale), "SDL Raymarching", nullptr, nullptr);
    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup scaling
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(main_scale);        // Bake a fixed style scale. (until we have a solution for dynamic style scaling, changing this requires resetting Style + calling this again)
    style.FontScaleDpi = main_scale;        // Set initial font scale. (in docking branch: using io.ConfigDpiScaleFonts=true automatically overrides this for every window depending on the current monitor)

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
#ifdef __EMSCRIPTEN__
    ImGui_ImplGlfw_InstallEmscriptenCallbacks(window, "#canvas");
#endif
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Build and cache SDF shader
    GLuint sdf_prog = create_program(g_vert_src, g_frag_src);
    GLint  loc_iTime       = glGetUniformLocation(sdf_prog, "iTime");
    GLint  loc_iMouse      = glGetUniformLocation(sdf_prog, "iMouse");
    GLint  loc_iResolution = glGetUniformLocation(sdf_prog, "iResolution");
    GLint  loc_iFrame      = glGetUniformLocation(sdf_prog, "iFrame");

    // Fullscreen quad (triangle strip: BL, BR, TL, TR)
    float quad_verts[] = { -1.0f, -1.0f,  1.0f, -1.0f,  -1.0f, 1.0f,  1.0f, 1.0f };
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_verts), quad_verts, GL_STATIC_DRAW);
    GLint pos_loc = glGetAttribLocation(sdf_prog, "position");
    glEnableVertexAttribArray(pos_loc);
    glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindVertexArray(0);

    int   frame      = 0;
    float start_time = (float)glfwGetTime();

    // Main loop
#ifdef __EMSCRIPTEN__
    // For an Emscripten build we are disabling file-system access, so let's not attempt to do a fopen() of the imgui.ini file.
    // You may manually call LoadIniSettingsFromMemory() to load settings from your own storage.
    io.IniFilename = nullptr;
    EMSCRIPTEN_MAINLOOP_BEGIN
#else
    while (!glfwWindowShouldClose(window))
#endif
    {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0)
        {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw SDF shader as fullscreen background
        {
            float itime = (float)glfwGetTime() - start_time;

            // GLFW cursor pos has origin top-left; flip Y for OpenGL (origin bottom-left)
            double mx, my;
            glfwGetCursorPos(window, &mx, &my);
            my = display_h - my;

            glUseProgram(sdf_prog);
            glUniform1f(loc_iTime,       itime);
            glUniform2f(loc_iMouse,      (float)mx, (float)my);
            glUniform3f(loc_iResolution, (float)display_w, (float)display_h, 1.0f);
            glUniform1i(loc_iFrame,      frame);

            glBindVertexArray(vao);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            glBindVertexArray(0);
            glUseProgram(0);
        }

        // Draw ImGui on top
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        ++frame;
    }
#ifdef __EMSCRIPTEN__
    EMSCRIPTEN_MAINLOOP_END;
#endif

    // Cleanup
    glDeleteProgram(sdf_prog);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
