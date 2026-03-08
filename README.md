# SDL_Raymarch_SDF

A real-time SDF raymarcher built with **GLFW + OpenGL 3 + Dear ImGui**, adapted from [Inigo Quilez's Shadertoy distance-function collection](https://iquilezles.org/articles/distfunctions/). The entire render pipeline runs inside a single GLSL fragment shader on a fullscreen quad.

---

## Architecture

```
main.cpp
├── Fullscreen quad (2 triangles) rendered via OpenGL
├── Vertex shader   – passthrough, maps clip-space [-1,1]² to gl_Position
└── Fragment shader – everything else:
    ├── SDF primitives (20+ shapes)
    ├── Scene map()  – CSG union of all primitives + bounding-box culls
    ├── raycast()    – sphere tracing loop
    ├── calcNormal() – tetrahedron finite-difference gradient
    ├── calcSoftshadow() – single-ray penumbra approximation
    ├── calcAO()     – normal-march ambient occlusion
    ├── render()     – 4-term shading model + fog
    └── mainImage()  – camera, ray differentials, 2×2 grid AA, gamma
```

Uniforms passed from C++: `iTime`, `iMouse`, `iResolution`, `iFrame`.

---

## 1. Signed Distance Functions

An SDF $f(\mathbf{p})$ returns the **signed shortest distance** from point $\mathbf{p}$ to the surface:

$$f(\mathbf{p}) \begin{cases} < 0 & \mathbf{p} \text{ is inside} \\ = 0 & \mathbf{p} \text{ is on the surface} \\ > 0 & \mathbf{p} \text{ is outside (exact distance)} \end{cases}$$

The key property that enables sphere tracing: **the ray can advance by exactly $f(\mathbf{p})$ without overshooting any surface**.

### 1.1 Utility helpers

```glsl
float dot2(vec3 v) { return dot(v,v); }         // squared length, avoids sqrt
float ndot(vec2 a, vec2 b) { return a.x*b.x - a.y*b.y; } // "diagonal dot" for rhombus
```

`dot2` avoids a `sqrt` when only comparing distances.
`ndot` projects onto the rhombus diagonal axes instead of the standard axes.

---

### 1.2 Primitive SDFs

#### Plane

$$f(\mathbf{p}) = p_y$$

The ground plane $y = 0$. The trivial SDF: distance from a point to a horizontal plane is just its height component.

```glsl
float sdPlane(vec3 p) { return p.y; }
```

---

#### Sphere

$$f(\mathbf{p}) = \|\mathbf{p}\| - r$$

```glsl
float sdSphere(vec3 p, float s) { return length(p) - s; }
```

The most fundamental SDF. All other SDFs are derived from this idea.

---

#### Box (axis-aligned)

$$\mathbf{d} = |\mathbf{p}| - \mathbf{b}$$

$$f(\mathbf{p}) = \underbrace{\left\|\max(\mathbf{d},\mathbf{0})\right\|}_{\text{exterior}} + \underbrace{\min\!\left(\max(d_x, d_y, d_z),\ 0\right)}_{\text{interior}}$$

**The corner trick:**
1. Fold $\mathbf{p}$ into the positive octant with $\mathrm{abs}()$, exploiting 3-axis symmetry.
2. $\mathbf{d} = |\mathbf{p}| - \mathbf{b}$ gives signed per-axis distances to the box faces. Negative components mean $\mathbf{p}$ is inside that slab.
3. $\|\max(\mathbf{d}, 0)\|$: exterior Euclidean distance (only positive components matter).
4. $\min(\max(d_x, d_y, d_z), 0)$: interior distance (largest negative = closest face).

Combined, this gives a **continuous exact SDF** everywhere.

```glsl
float sdBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}
```

---

#### Box Frame (wireframe)

Applies `abs()` twice with different offsets to fold $\mathbf{p}$ into the **corner region**. Each of the 3 terms covers one pair of parallel edges (x-edges, y-edges, z-edges); `min()` selects the closest edge.

```glsl
float sdBoxFrame(vec3 p, vec3 b, float e) {
    p = abs(p) - b;
    vec3 q = abs(p + e) - e;
    return min(min(
        length(max(vec3(p.x, q.y, q.z), 0.0)) + min(max(p.x, max(q.y, q.z)), 0.0),
        length(max(vec3(q.x, p.y, q.z), 0.0)) + min(max(q.x, max(p.y, q.z)), 0.0)),
        length(max(vec3(q.x, q.y, p.z), 0.0)) + min(max(q.x, max(q.y, p.z)), 0.0));
}
```

---

#### Ellipsoid (approximate)

$$k_0 = \left\|\frac{\mathbf{p}}{\mathbf{r}}\right\|, \quad k_1 = \left\|\frac{\mathbf{p}}{\mathbf{r}^2}\right\|, \quad f(\mathbf{p}) \approx \frac{k_0(k_0 - 1)}{k_1}$$

**Not exact Euclidean distance** — $k_0$ normalizes the ellipsoid to a unit sphere; $k_1$ is proportional to the gradient magnitude and corrects for non-uniform scaling. The approximation slightly overestimates, keeping sphere tracing **conservative (safe)**.

---

#### Torus

Reduce to 2D: distance from the torus axis circle, then subtract tube radius.

$$f(\mathbf{p}) = \left\|\begin{pmatrix}\|\mathbf{p}_{xz}\| - t_x \\ p_y\end{pmatrix}\right\| - t_y$$

where $t_x$ = major radius, $t_y$ = tube radius.

```glsl
float sdTorus(vec3 p, vec2 t) {
    return length(vec2(length(p.xz) - t.x, p.y)) - t.y;
}
```

---

#### Capped Torus

A torus with a wedge cut by an opening angle. `sc = (sin θ, cos θ)` of the half-angle.

**Trick:** fold the XY plane so the cut is at $x=0$, then clamp $k$ to the arc or project onto the endpoint cap.

$$f(\mathbf{p}) = \sqrt{\mathbf{p}\cdot\mathbf{p} + r_a^2 - 2 r_a k} - r_b$$

where $k = \begin{cases} \mathbf{p}_{xy} \cdot \mathbf{sc} & \text{if in the arc} \\ \|\mathbf{p}_{xy}\| & \text{if in the gap (cap region)} \end{cases}$

---

#### Capsule

Project $\mathbf{p}$ onto segment $\overline{AB}$, clamp, subtract radius.

$$h = \mathrm{clamp}\!\left(\frac{(\mathbf{p}-\mathbf{a}) \cdot (\mathbf{b}-\mathbf{a})}{\|\mathbf{b}-\mathbf{a}\|^2},\ 0,\ 1\right)$$

$$f(\mathbf{p}) = \left\|\mathbf{p} - \mathbf{a} - (\mathbf{b}-\mathbf{a})\,h\right\| - r$$

---

#### Cylinder (vertical)

Same box trick but in cylindrical coordinates:

$$\mathbf{d} = \begin{pmatrix}|\,\|\mathbf{p}_{xz}\|\,| - h_r \\ |p_y| - h_h\end{pmatrix}$$

$$f(\mathbf{p}) = \min(\max(d_x, d_y), 0) + \|\max(\mathbf{d}, 0)\|$$

Gives correct rounded edges at the cap-wall junction.

---

#### Cone

Reduce to 2D $(w_x = \|\mathbf{p}_{xz}\|,\ w_y = p_y)$, project onto the slant edge and base cap, take minimum:

$$f = \sqrt{\min\!\left(\|\mathbf{w} - \mathbf{q}\,t_a\|^2,\ \|\mathbf{w} - \mathbf{q}\,t_b\|^2\right)} \cdot \mathrm{sign}(s)$$

where $t_a, t_b$ are clamped parametric projections onto the slant and base respectively.

---

#### Solid Angle

Intersection of a sphere (radius $r_a$) and an infinite cone with half-angle $\theta$ (given as $\mathbf{c} = (\sin\theta, \cos\theta)$):

$$l = \|\mathbf{p}_{xz \to 2D}\| - r_a \quad \text{(sphere distance)}$$
$$m = \|\mathbf{p}_{2D} - \mathbf{c}\,\mathrm{clamp}(\mathbf{p}_{2D}\cdot\mathbf{c},\ 0,\ r_a)\| \quad \text{(cone edge distance)}$$

$$f = \max\!\left(l,\ m\cdot\mathrm{sign}(c_y p_x - c_x p_y)\right)$$

---

#### Octahedron

$|x|+|y|+|z| = s$ is a plane in the folded positive octant. After folding with `abs()`, classify which face $\mathbf{p}$ projects onto using the L1 norm $m = p_x + p_y + p_z - s$:

$$f(\mathbf{p}) = \begin{cases} m / \sqrt{3} & \text{corner region} \\ \|\mathbf{q}\| & \text{face region, after clamping} \end{cases}$$

where $1/\sqrt{3} \approx 0.57735027$ is the distance scale for the face normal.

---

#### Hex / Octagon Prism

**Reflection trick:** reflect $\mathbf{p}$ into the fundamental domain using the polygon's symmetry normals. For a hexagon: 2 reflections with $\mathbf{k}_{xy} = (-\sqrt{3}/2,\ 0.5)$ collapse 6-fold symmetry to a 30° wedge. For an octagon: 2 reflections at 22.5° and 67.5°. Then compute distance to nearest flat edge + cap distance using the box formula.

---

#### Pyramid

Exploit 4-fold symmetry (`abs(p.xz)` + sort), then project onto a triangular face. The face normal magnitude is $m_2 = h^2 + 0.25$.

Two candidate distances $a, b$ cover the two edges of the triangular face; take the minimum and correct for scale:

$$f = \frac{\sqrt{\min(a,b) + q_z^2}}{m_2} \cdot \mathrm{sign}(\max(q_z, -p_y))$$

---

#### Rhombus

`ndot` projects onto the rhombus diagonal axes:

$$f_{\text{rhombus}}(\mathbf{p}) = \mathrm{boxSDF2D}\!\left(\,\|\mathbf{p}_{xz} - \tfrac{1}{2}\mathbf{b}(1-f, 1+f)\| - r_a,\ p_y - h\right)$$

where $f = \mathrm{clamp}\!\left(\frac{\mathrm{ndot}(\mathbf{b}, \mathbf{b} - 2\mathbf{p}_{xz})}{\mathbf{b}\cdot\mathbf{b}},\ -1,\ 1\right)$ is the perimeter parameter.

---

### 1.3 Boolean CSG Operations

```glsl
vec2 opU(vec2 d1, vec2 d2) { return (d1.x < d2.x) ? d1 : d2; }
```

`opU` = **union**: `min(d1, d2)` takes the nearer surface. The `vec2` carries `(distance, material_id)` together, so material information propagates for free.

Other standard CSG ops (not in this scene, but common):

| Operation | Formula |
|---|---|
| Union | $\min(d_1, d_2)$ |
| Intersection | $\max(d_1, d_2)$ |
| Subtraction | $\max(d_1, -d_2)$ |
| Smooth union | $d_1 - h^2/(4k)$ with $h = \max(k - \|d_1 - d_2\|, 0)$ |

---

## 2. Sphere Tracing (Raymarching)

**Algorithm** for ray $\mathbf{r}(t) = \mathbf{o} + t\,\mathbf{d}$:

$$t_{n+1} = t_n + f(\mathbf{o} + t_n\,\mathbf{d})$$

The SDF guarantees no surface exists within a sphere of radius $f(\mathbf{p})$ centered at $\mathbf{p}$, so advancing by $f(\mathbf{p})$ **never overshoots**.

**Termination:**

$$|f(\mathbf{p})| < \varepsilon_{\text{rel}} \cdot t \quad \text{(hit)}$$

The **relative epsilon** $\varepsilon_{\text{rel}} = 10^{-4}$ allows more tolerance at distance (where one pixel covers more world space), preventing false hits at grazing angles on flat surfaces.

```glsl
for (int i = 0; i < 70 && t < tmax; i++) {
    vec2 h = map(ro + rd * t);
    if (abs(h.x) < 0.0001 * t) { res = vec2(t, h.y); break; }
    t += h.x;
}
```

**70 steps** is sufficient for this scene. Thin features or complex Boolean operations may need more.

---

### 2.1 Ray-AABB Broad Phase (`iBox`)

Before sphere tracing, intersect the ray with the bounding box of all primitives. Only march inside $[t_\text{enter}, t_\text{exit}]$, skipping all empty space outside.

**Slab method derivation:** for each axis, the ray crosses the two slab planes at:

$$t = \frac{\pm r - o_i}{d_i} = -m_i \pm k_i, \quad m_i = \frac{o_i}{d_i},\quad k_i = \frac{r_i}{|d_i|}$$

$$t_\text{enter} = \max(t_{x,\text{min}},\, t_{y,\text{min}},\, t_{z,\text{min}})$$
$$t_\text{exit}  = \min(t_{x,\text{max}},\, t_{y,\text{max}},\, t_{z,\text{max}})$$

Hit iff $t_\text{enter} < t_\text{exit}$.

```glsl
vec2 iBox(vec3 ro, vec3 rd, vec3 rad) {
    vec3 m = 1.0 / rd;
    vec3 n = m * ro;
    vec3 k = abs(m) * rad;
    return vec2(max(max(-n.x-k.x, -n.y-k.y), -n.z-k.z),
                min(min(-n.x+k.x, -n.y+k.y), -n.z+k.z));
}
```

---

### 2.2 Per-Group Bounding Box Culling

Each group of scene objects is wrapped in a cheap `sdBox` test inside `map()`:

```glsl
if (sdBox(pos - groupCenter, groupHalfExtents) < res.x) {
    // evaluate expensive SDFs only if bounding box is closer than current best
}
```

**Why it's valid:** the SDF is a lower bound — if the bounding box distance exceeds the current best hit, no primitive inside can be closer. This is an **O(1) rejection per group**, equivalent to a 1-level BVH without any data structure overhead.

---

### 2.3 The `ZERO` Trick (Shader Compiler Anti-Unrolling)

```glsl
#define ZERO (min(iFrame, 0))
```

`iFrame` is a runtime uniform. The compiler **cannot prove** `min(iFrame, 0) == 0` at compile time, so it cannot statically unroll loops that start at `ZERO`. Unrolling massively bloats the shader binary and hurts GPU register pressure, increasing register spilling and reducing warp occupancy.

All loops that call `map()` use `ZERO` as the start index.

---

## 3. Normal Estimation — Tetrahedron Gradient

The gradient of an SDF equals the surface normal: $\mathbf{n} = \nabla f(\mathbf{p})$.

**Naïve finite differences** use 6 `map()` calls (±ε per axis). **IQ's tetrahedron trick** uses only **4 calls** with a regular tetrahedron stencil inscribed in a cube:

$$\mathbf{e}_0 = (+1,+1,-1),\quad \mathbf{e}_1 = (-1,-1,-1),\quad \mathbf{e}_2 = (-1,+1,+1),\quad \mathbf{e}_3 = (+1,-1,+1)$$

$$\mathbf{n} = \mathrm{normalize}\!\left(\sum_{i=0}^{3} \mathbf{e}_i \cdot f(\mathbf{p} + \varepsilon\,\mathbf{e}_i)\right)$$

The cross-terms cancel exactly (each axis appears with equal $+$ and $-$ weight), giving the true gradient. $0.5773 \approx 1/\sqrt{3}$ normalizes the tetrahedron vertices to unit length. $\varepsilon = 0.0005$ balances accuracy vs. float precision.

**Cost: 4 `map()` calls instead of 6 — a 33% saving at the hottest spot in the pipeline.**

```glsl
vec3 n = vec3(0.0);
for (int i = ZERO; i < 4; i++) {
    vec3 e = 0.5773 * (2.0 * vec3(((i+3)>>1)&1, (i>>1)&1, i&1) - 1.0);
    n += e * map(pos + 0.0005 * e).x;
}
return normalize(n);
```

---

## 4. Soft Shadows

**Trick:** instead of Monte Carlo area-light sampling ($O(N^2)$ rays), march a **single shadow ray** toward the light and track the closest approach to any occluder.

At each step $t$ along the shadow ray, $h = f(\mathbf{p})$ is the distance to the nearest surface. The ratio:

$$s_i = \frac{h}{t} \approx \sin(\alpha)$$

where $\alpha$ is the angular gap between the shadow ray and the nearest occluder. Scaled by $k$ (penumbra width) and clamped:

$$\text{penumbra} = \min_i\!\left(\mathrm{clamp}(k \cdot h_i / t_i,\ 0,\ 1)\right)$$

The minimum tracks the **darkest point** along the ray (deepest into shadow). Final smoothstep:

$$\text{shadow} = s^2(3 - 2s)$$

$$\text{Cost: 24 map() calls vs. } O(N^2) \text{ for a true area light.}$$

```glsl
float res = 1.0, t = mint;
for (int i = ZERO; i < 24; i++) {
    float h = map(ro + rd * t).x;
    float s = clamp(8.0 * h / t, 0.0, 1.0);
    res = min(res, s);
    t += clamp(h, 0.01, 0.2);
    if (res < 0.004 || t > tmax) break;
}
return res * res * (3.0 - 2.0 * res);
```

---

## 5. Ambient Occlusion

**Trick:** instead of integrating visibility over the hemisphere:

$$\text{AO} = \frac{1}{\pi} \int_\Omega V(\boldsymbol{\omega})\,(\boldsymbol{\omega}\cdot\mathbf{n})\,d\omega$$

march **5 samples along the surface normal** at increasing distances:

$$\text{occ} = \sum_{i=0}^{4} (h_i - f(\mathbf{p} + h_i\,\mathbf{n})) \cdot w_i, \quad w_i = 0.95^i$$

At distance $h_i$ along the normal, the SDF should equal $h_i$ if space is open. If $f < h_i$, the surface curves back and occludes: the shortfall $(h_i - f)$ accumulates. Exponential weights give more importance to **closer occluders**.

$$\text{AO} = \mathrm{clamp}(1 - 3\,\text{occ},\ 0,\ 1) \cdot (0.5 + 0.5\,n_y)$$

The $(0.5 + 0.5\,n_y)$ term gives a directional bias: surfaces facing up are more exposed to the sky and get less AO reduction.

**Cost: 5 map() calls vs. 100+ for Monte Carlo hemisphere AO.**

---

## 6. Shading Model

The full rendering equation:

$$L_o(\mathbf{p}, \boldsymbol{\omega}_o) = \int_\Omega f_r(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o)\,L_i(\mathbf{p}, \boldsymbol{\omega}_i)\,(\boldsymbol{\omega}_i\cdot\mathbf{n})\,d\omega_i$$

is **not solved** — it is replaced by 4 analytical heuristic terms:

### 6.1 Sun (directional light)

$$L_\text{diffuse} = \mathbf{c} \cdot 2.2 \cdot \max(\mathbf{n}\cdot\mathbf{l}, 0) \cdot \text{shadow} \cdot (1.3, 1.0, 0.7)$$

$$L_\text{specular} = 5.0 \cdot \max(\mathbf{n}\cdot\mathbf{h}, 0)^{16} \cdot F(\mathbf{h},\mathbf{l}) \cdot \text{shadow}$$

**Blinn-Phong halfway vector:** $\mathbf{h} = \mathrm{normalize}(\mathbf{l} - \mathbf{d})$, cheaper than computing the reflection direction.

**Schlick Fresnel approximation:**

$$F(\mathbf{h}, \mathbf{l}) \approx F_0 + (1 - F_0)(1 - \mathbf{h}\cdot\mathbf{l})^5$$

with $F_0 = 0.04$ (normal-incidence reflectance of typical dielectrics — glass, plastic). At grazing angles $F \to 1$; at normal incidence $F \to 0.04$.

### 6.2 Sky (ambient / indirect)

Approximation of $\int_\text{upper hemisphere} L_\text{sky}\,(\boldsymbol{\omega}\cdot\mathbf{n})\,d\omega$ for a uniform blue sky:

$$L_\text{sky} = \mathbf{c} \cdot 0.6 \cdot \sqrt{\max(0.5 + 0.5\,n_y, 0)} \cdot \text{AO} \cdot (0.4, 0.6, 1.15)$$

`sqrt` softens the falloff: sky light looks brighter overhead but with a gradual roll-off, not the sharp Lambertian $\cos\theta$.

**Sky specular:** if the reflection direction $\mathbf{r}$ points into the sky:

$$L_\text{sky,spec} = 2.0 \cdot \text{smoothstep}(-0.2, 0.2, r_y) \cdot F(\mathbf{n}, \mathbf{d}) \cdot \text{shadow}(\mathbf{r})$$

### 6.3 Back Light (fill / bounce)

A second fake directional light from the opposite side of the sun. Simulates light bounced off the ground or a back wall. Attenuated with height so only low objects get fill light:

$$L_\text{fill} = \mathbf{c} \cdot 0.55 \cdot \max(\mathbf{n}\cdot\mathbf{l}_\text{back}, 0) \cdot \max(1 - p_y, 0) \cdot \text{AO} \cdot (0.25, 0.25, 0.25)$$

### 6.4 SSS / Rim Light

Fakes subsurface scattering / translucency (wax, skin, leaves). Strongest at **grazing/backlit angles**:

$$L_\text{rim} = \mathbf{c} \cdot 0.25 \cdot \max(1 + \mathbf{n}\cdot\mathbf{d},\ 0)^2 \cdot \text{AO}$$

$(1 + \mathbf{n}\cdot\mathbf{d})$ is 0 on front faces (normal faces away from view), maximum at grazing/back-lit angles.

---

## 7. Atmospheric Fog

Cubic exponential depth fog:

$$\mathbf{c}_\text{final} = \mathrm{mix}(\mathbf{c},\ (0.7, 0.7, 0.9),\ 1 - e^{-k\,t^3})$$

$t^3$ makes the onset very gradual close up, then rapid far away. The sky color $(0.7, 0.7, 0.9)$ matches the background so objects smoothly fade into it at distance.

---

## 8. Camera and Ray Differentials

**Camera basis (look-at):** given eye $\mathbf{o}$ and target $\mathbf{t}$:

$$\mathbf{c}_w = \mathrm{normalize}(\mathbf{t} - \mathbf{o}), \quad \mathbf{c}_u = \mathrm{normalize}(\mathbf{c}_w \times \mathbf{c}_p), \quad \mathbf{c}_v = \mathbf{c}_u \times \mathbf{c}_w$$

where $\mathbf{c}_p = (\sin\theta_\text{roll},\, \cos\theta_\text{roll},\, 0)$ is the up-reference with optional roll angle.

**Primary ray direction** (perspective projection):

$$\mathbf{d} = \mathrm{normalize}\!\left(M_\text{cam} \cdot \begin{pmatrix} p_x \\ p_y \\ f_l \end{pmatrix}\right), \quad f_l = 2.5 \Rightarrow \text{FOV} \approx 44°$$

where $\mathbf{p} = \frac{2\,\text{fragCoord} - \text{resolution}}{h}$ maps pixels to NDC with square-pixel aspect ratio.

**Ray differentials** for the floor anti-aliasing: rays to neighboring pixels $(x+1,\, y+1)$ define the **pixel footprint** in world space:

```glsl
vec3 rdx = ca * normalize(vec3(px, fl));  // ray to pixel (x+1, y)
vec3 rdy = ca * normalize(vec3(py, fl));  // ray to pixel (x, y+1)
```

On the floor plane ($y = 0$), the differential footprint is:

$$\frac{\partial \mathbf{p}}{\partial x} = \frac{o_y}{d_y}\left(\frac{\mathbf{d}}{d_y} - \frac{\mathbf{d}_x}{d_{x,y}}\right)$$

This $\partial\mathbf{p}/\partial x$ and $\partial\mathbf{p}/\partial y$ express how much world position shifts per screen pixel, used to analytically filter the checkerboard pattern.

---

## 9. Anti-Aliased Checkerboard (`checkersGradBox`)

**Trick:** instead of point-sampling the checkerboard (which aliases at a distance), analytically **integrate it over the pixel footprint**.

The box filter of a sign-wave checkerboard over a footprint $[p - w/2,\, p + w/2]$ has a closed form:

$$I(p, w) = 2\!\left(\left|\mathrm{fract}\!\left(\frac{p - w/2}{2}\right) - \tfrac{1}{2}\right| - \left|\mathrm{fract}\!\left(\frac{p + w/2}{2}\right) - \tfrac{1}{2}\right|\right) / w$$

The 2D XOR pattern is:

$$f = 0.5 - 0.5\,I_x \cdot I_y$$

where $w = |\partial p / \partial x| + |\partial p / \partial y| + \epsilon$ is the L1 footprint size. This is equivalent to hardware mipmapping for an infinite procedural pattern: **no Moiré, no shimmer, no aliasing at any distance**.

---

## 10. Grid Supersampling (AA)

```glsl
#define AA 2  // 2×2 = 4 samples/pixel
```

For each pixel, evaluate $\text{AA}^2$ evenly spaced subpixel samples at offsets $o = (m, n) / \text{AA} - 0.5$ for $m, n \in \{0, \ldots, \text{AA}-1\}$, then average:

$$C_\text{pixel} = \frac{1}{\text{AA}^2} \sum_{m,n} C(p + o_{m,n})$$

Unlike **Monte Carlo AA** (random samples), grid supersampling is **deterministic and noise-free**: every frame produces the same result. The tradeoff is that high-frequency patterns aliasing between fixed sample positions are not captured, but for this scene 4 samples is sufficient.

On slow GPUs, define `HW_PERFORMANCE 0` to fall back to 1 sample/pixel.

---

## 11. Gamma Correction

All lighting is computed in **linear light space**. Monitor displays use sRGB (approximately $\gamma = 2.2$). Converting to display-ready output:

$$C_\text{sRGB} \approx C_\text{linear}^{1/2.2} = C_\text{linear}^{0.4545}$$

```glsl
col = pow(col, vec3(0.4545));
```

Without this step, the midtones appear too bright and washed out.

---

## Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./example_glfw_opengl3
```

Dependencies: `libglfw3-dev`, `libgl-dev` (OpenGL). Dear ImGui and its GLFW/OpenGL3 backends are vendored in `vendor/imgui/`.

---

## References

- Inigo Quilez, [*Distance Functions*](https://iquilezles.org/articles/distfunctions/) — source of all SDF primitives
- Inigo Quilez, [*Sphere Tracing*](https://iquilezles.org/articles/raymarchingdf/) — the core algorithm
- Inigo Quilez, [*Soft Shadows*](https://iquilezles.org/articles/rmshadows/) — the `h/t` penumbra trick
- Hart, J.C. (1996), *Sphere Tracing: A Geometric Method for the Antialiased Ray Tracing of Implicit Surfaces*
- Christoph Peters, [*Checkerboard anti-aliasing*](https://gpuopen.com/learn/optimized-reversible-tonemapper-for-resolve/) — analytical box filter
