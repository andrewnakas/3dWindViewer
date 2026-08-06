// GLSL for the GPU particle system (WebGL2 / GLSL ES 3.00).

export const QUAD_VERT = `#version 300 es
precision highp float;
out vec2 v_uv;
void main() {
  // fullscreen triangle
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  v_uv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

// Advect particle state. State texel: R,G = pos (0..1 across data bounds,
// y=0 at NORTH edge to match atlas row order), B = age, A = normalized speed.
export const UPDATE_FRAG = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outState;

uniform sampler2D u_state;
uniform sampler2D u_frameA;
uniform sampler2D u_frameB;
uniform float u_frameMix;

uniform vec2 u_tileOffset;   // atlas UV of tile origin
uniform vec2 u_tileScale;    // tile extent in atlas UV
uniform vec2 u_clampMin;     // half-texel clamp inside tile (tile-local 0..1)
uniform vec2 u_clampMax;
uniform vec2 u_uRange;       // uMin, uMax (m/s)
uniform vec2 u_vRange;
uniform float u_west;
uniform float u_north;
uniform float u_lonSpan;
uniform float u_latSpan;
uniform float u_dt;          // simulated seconds per rendered frame
uniform float u_maxAge;
uniform float u_speedMax;    // for color normalization
uniform float u_time;        // random seed

float rand(vec2 co) {
  return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
  vec4 st = texture(u_state, v_uv);
  vec2 pos = st.rg;
  float age = st.b;

  vec2 tuv = u_tileOffset + clamp(pos, u_clampMin, u_clampMax) * u_tileScale;
  vec4 w = mix(texture(u_frameA, tuv), texture(u_frameB, tuv), u_frameMix);
  float u = mix(u_uRange.x, u_uRange.y, w.r);
  float v = mix(u_vRange.x, u_vRange.y, w.g);

  float lat = u_north - pos.y * u_latSpan;
  float dlon = u * u_dt / (111320.0 * max(cos(radians(lat)), 0.05));
  float dlat = v * u_dt / 110540.0;
  vec2 npos = pos + vec2(dlon / u_lonSpan, -dlat / u_latSpan);
  age += 1.0;

  float speedN = clamp(length(vec2(u, v)) / u_speedMax, 0.0, 1.0);

  float lifetime = u_maxAge * (0.5 + rand(v_uv * 7.13));
  bool dead = age > lifetime
    || npos.x < 0.0 || npos.x > 1.0 || npos.y < 0.0 || npos.y > 1.0
    || w.a < 0.5;
  if (dead) {
    npos = vec2(rand(v_uv + fract(u_time)), rand(v_uv.yx + fract(u_time) * 1.71));
    age = 0.0;
  }
  outState = vec4(npos, age, speedN);
}`;

// Draw particles as short streak segments (GL_LINES, 2 verts per particle).
export const DRAW_VERT = `#version 300 es
precision highp float;

uniform sampler2D u_stateCurr;
uniform sampler2D u_statePrev;
uniform int u_stateSize;
uniform mat4 u_matrix;
uniform float u_west;
uniform float u_north;
uniform float u_lonSpan;
uniform float u_latSpan;
uniform float u_alt;       // altitude in mercator z units (premultiplied)
uniform float u_streak;    // tail extension factor
uniform float u_maxAge;

out float v_speed;
out float v_alpha;

const float PI = 3.141592653589793;

void main() {
  int pid = gl_VertexID / 2;
  int end = gl_VertexID - pid * 2;
  ivec2 tc = ivec2(pid % u_stateSize, pid / u_stateSize);
  vec4 sc = texelFetch(u_stateCurr, tc, 0);
  vec4 sp = texelFetch(u_statePrev, tc, 0);

  vec2 pc = sc.rg;
  vec2 pp = sp.rg;
  if (distance(pc, pp) > 0.02) pp = pc;  // collapse respawn jumps
  vec2 pos = (end == 0) ? pc + (pp - pc) * u_streak : pc;

  float lon = u_west + pos.x * u_lonSpan;
  float lat = u_north - pos.y * u_latSpan;
  float mx = (lon + 180.0) / 360.0;
  float s = clamp(sin(radians(lat)), -0.9999, 0.9999);
  float my = 0.5 - 0.25 * log((1.0 + s) / (1.0 - s)) / PI;

  gl_Position = u_matrix * vec4(mx, my, u_alt, 1.0);

  v_speed = sc.a;
  float fadeIn = clamp(sc.b / 10.0, 0.0, 1.0);
  float fadeOut = 1.0 - smoothstep(0.6, 1.0, sc.b / u_maxAge);
  float endDim = (end == 0) ? 0.1 : 1.0;
  v_alpha = fadeIn * fadeOut * endDim;
}`;

export const DRAW_FRAG = `#version 300 es
precision highp float;

in float v_speed;
in float v_alpha;
out vec4 outColor;

uniform sampler2D u_ramp;
uniform float u_opacity;

void main() {
  vec3 color = texture(u_ramp, vec2(v_speed, 0.5)).rgb;
  outColor = vec4(color, 1.0) * (v_alpha * u_opacity);
}`;
