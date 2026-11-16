// Interactive CMA-ES-IG demo: rank Jaco2 handovers, entirely in the browser.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import URDFLoader from "urdf-loader";
import { ALGORITHMS, ParticleBelief, RNG, standardise } from "./cmaesig.js";

const ROOT = new URL("..", import.meta.url).href; // .../static/
const N_TRAJ = 1000, T = 50, DOF = 6, FEAT = 8;
const K = 3;
const LANE_SPACING = 1.4;
const LANES = [
  { name: "A", color: "#0072b2" },
  { name: "B", color: "#d55e00" },
  { name: "C", color: "#009e73" },
];
const BEST_COLOR = "#e69f00";
const REAL_DURATION = 9; // seconds per handover on the real robot
const HOLD = 1.2;
const ARM_JOINTS = [1, 2, 3, 4, 5, 6].map((i) => `j2n6s300_joint_${i}`);
// Finger angles and cup offset chosen so the fingertips touch the cup without intersecting it.
const GRIP = { finger: 0.7, finger_tip: 0.1 };
const EE_LINK = "j2n6s300_end_effector";
const PLACES = ["1st", "2nd", "3rd"];
const CAPTIONS = {
  "CMA-ES-IG": "CMA-ES-IG draws 100 samples from its search distribution, clusters them with K-means, and shows you the handovers nearest the 3 centroids.",
  "CMA-ES": "CMA-ES samples the 3 candidates directly from its search distribution, so they are often close together and harder to tell apart.",
  Infogain: "Infogain picks the 3 handovers whose ranking would tell the robot the most about your preference. It has no search distribution, so it doesn't steer toward what you like. (To keep the demo responsive, this is an approximation: it scores 300 random triples instead of running the full optimisation used in the paper.)",
};

const $ = (sel, el = document) => el.querySelector(sel);

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------
async function loadF32(path) {
  const res = await fetch(ROOT + path);
  if (!res.ok) throw new Error(`Could not load ${path}`);
  return new Float32Array(await res.arrayBuffer());
}


// Catmull-Rom interpolation of a (T x DOF) trajectory at phase in [0, 1].
function sampleTraj(traj, phase, out) {
  const u = Math.min(Math.max(phase, 0), 1) * (T - 1);
  const i = Math.min(Math.floor(u), T - 2), t = u - i;
  const at = (k, j) => traj[Math.min(Math.max(k, 0), T - 1) * DOF + j];
  for (let j = 0; j < DOF; j++) {
    const p0 = at(i - 1, j), p1 = at(i, j), p2 = at(i + 1, j), p3 = at(i + 2, j);
    out[j] = 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

// Draw calls, not triangles, dominate on some GPUs (notably WebGL on Metal), so
// meshes under `root` that share a material are merged into one mesh each.
// Subtrees starting at an object in `stops` (moving joints) are left alone so
// they can be merged separately. Only positions and normals are kept.
function mergeStatic(root, stops = new Set()) {
  root.updateMatrixWorld(true);
  const toRoot = root.matrixWorld.clone().invert();
  const found = [];
  (function visit(o) {
    for (const c of o.children) {
      if (stops.has(c)) continue;
      if (c.isMesh) found.push(c);
      visit(c);
    }
  })(root);
  const byMaterial = new Map();
  for (const m of found) {
    const g = m.geometry.clone().applyMatrix4(new THREE.Matrix4().multiplyMatrices(toRoot, m.matrixWorld));
    for (const name of Object.keys(g.attributes)) if (name !== "position" && name !== "normal") g.deleteAttribute(name);
    if (!byMaterial.has(m.material)) byMaterial.set(m.material, { geoms: [], src: m });
    byMaterial.get(m.material).geoms.push(g);
  }
  for (const m of found) m.removeFromParent();
  for (const [material, { geoms, src }] of byMaterial) {
    const indexed = geoms.every((g) => g.index);
    const merged = new THREE.Mesh(mergeGeometries(indexed ? geoms : geoms.map((g) => (g.index ? g.toNonIndexed() : g))), material);
    merged.renderOrder = src.renderOrder;
    merged.userData = { ...src.userData };
    root.add(merged);
  }
}
function makeMaterials() {
  return {
    carbon: new THREE.MeshLambertMaterial({ color: 0x2a2c30 }),
    ring: new THREE.MeshLambertMaterial({ color: 0xb9bcc2 }),
    finger: new THREE.MeshLambertMaterial({ color: 0x3c3f45 }),
  };
}

function loadRobot(materials) {
  return new Promise((resolve, reject) => {
    const manager = new THREE.LoadingManager();
    const loader = new URDFLoader(manager);
    // Meshes are low-poly GLB versions of the kinova-ros Collada files.
    const gltf = new GLTFLoader(manager);
    const loadMesh = (path, mgr, cb) => /\.glb$/i.test(path)
      ? gltf.load(path, (g) => cb(g.scene), undefined, (e) => cb(null, e))
      : loader.defaultMeshLoader(path, mgr, cb);
    loader.loadMeshCb = (path, mgr, done) => {
      loadMesh(path, mgr, (obj, err) => {
        if (obj) {
          const mat = /ring/.test(path) ? materials.ring : /finger/.test(path) ? materials.finger : materials.carbon;
          obj.traverse((c) => {
            if (c.isMesh) c.material = mat;
          });
        }
        done(obj, err);
      });
    };
    let robot;
    manager.onLoad = () => resolve(robot);
    manager.onError = (url) => reject(new Error(`Failed to load ${url}`));
    loader.load(ROOT + "robot/j2n6s300/j2n6s300.urdf", (r) => { robot = r; }, undefined, reject);
  });
}

function setArm(robot, q) {
  for (let j = 0; j < DOF; j++) robot.setJointValue(ARM_JOINTS[j], q[j]);
}

function closeGripper(robot) {
  for (let f = 1; f <= 3; f++) {
    robot.setJointValue(`j2n6s300_joint_finger_${f}`, GRIP.finger);
    robot.setJointValue(`j2n6s300_joint_finger_tip_${f}`, GRIP.finger_tip);
  }
}

function makeCup() {
  const g = new THREE.Group();
  const mat = new THREE.MeshLambertMaterial({ color: 0x2f6fd0, side: THREE.DoubleSide });
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.03, 0.1, 32, 1, true), mat);
  const bottom = new THREE.Mesh(new THREE.CircleGeometry(0.03, 32), mat);
  bottom.rotation.x = Math.PI / 2;
  bottom.position.y = -0.05;
  g.add(body, bottom);
  // three.js cylinders run along +Y, across the gripper; +Z is the approach direction
  // and the palm ends at z = -0.039, so the cup sits just in front of it.
  g.position.set(0, 0, 0.02);
  return g;
}

function makeTable(group) {
  const top = new THREE.MeshLambertMaterial({ color: 0x8a6d55 });
  const leg = new THREE.MeshLambertMaterial({ color: 0x5c4838 });
  const table = new THREE.Group();
  const t = new THREE.Mesh(new THREE.BoxGeometry(0.95, 0.9, 0.04), top);
  t.position.set(0.3, 0.05, -0.02);
  table.add(t);
  for (const dx of [-0.12, 0.72]) for (const dy of [-0.34, 0.44]) {
    const l = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.04, 0.7), leg);
    l.position.set(dx, dy, -0.39);
    table.add(l);
  }
  mergeStatic(table);
  group.add(table);
}

// Minimum-jerk time scaling: the robot starts and stops at rest (zero velocity
// and acceleration) while following the same path.
const retime = (t) => t * t * t * (10 - 15 * t + 6 * t * t);

// A translucent seated person. They stay put while the robot rig is rotated.
// Seated where the (rotated, shifted) handovers end, around x = 0.67, y = -0.52, z = 0.30.
const PERSON = { x: 1.02, y: -0.52 };
const RIG_OFFSET = 0.35; // robot + table slide this far toward the person
function makePerson(group) {
  const ghost = new THREE.MeshLambertMaterial({ color: 0x7d8fa8, transparent: true, opacity: 0.5, depthWrite: false });
  const chairMat = new THREE.MeshLambertMaterial({ color: 0x5a5f66, transparent: true, opacity: 0.45, depthWrite: false });
  const person = new THREE.Group();
  person.position.set(PERSON.x, PERSON.y, 0);
  const add = (geom, mat, x, y, z) => { const m = new THREE.Mesh(geom, mat); m.position.set(x, y, z); person.add(m); return m; };
  // limb between two points (in the person's frame)
  const limb = (a, b, r) => {
    const A = new THREE.Vector3(...a), B = new THREE.Vector3(...b);
    const m = add(new THREE.CapsuleGeometry(r, A.distanceTo(B), 6, 12), ghost, ...A.clone().add(B).multiplyScalar(0.5).toArray());
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), B.clone().sub(A).normalize());
  };
  // chair
  add(new THREE.BoxGeometry(0.42, 0.42, 0.04), chairMat, 0.05, 0, -0.31);
  add(new THREE.BoxGeometry(0.04, 0.42, 0.45), chairMat, 0.25, 0, -0.08);
  for (const dx of [-0.13, 0.23]) for (const dy of [-0.18, 0.18]) add(new THREE.BoxGeometry(0.03, 0.03, 0.41), chairMat, dx, dy, -0.535);
  // body, facing the robot (-x)
  limb([0.06, 0, -0.2], [0.04, 0, 0.12], 0.13);                     // torso
  add(new THREE.SphereGeometry(0.095, 24, 16), ghost, 0.02, 0, 0.36); // head
  for (const dy of [-0.09, 0.09]) {
    limb([0.0, dy, -0.25], [-0.32, dy, -0.25], 0.055);              // thighs
    limb([-0.34, dy, -0.27], [-0.36, dy, -0.7], 0.045);             // shins
  }
  for (const side of [-1, 1]) limb([0.02, 0.17 * side, 0.12], [0.08, 0.21 * side, -0.12], 0.04); // arms at sides
  person.traverse((c) => { c.renderOrder = 1; });
  mergeStatic(person);
  const el = document.createElement("div");
  el.className = "you-label";
  el.textContent = "You";
  const label = new CSS2DObject(el);
  label.position.set(0.02, 0, 0.55);
  person.add(label);
  group.add(person);
  return person;
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------
export async function initDemo(container) {
  const viewport = $(".demo-viewport", container);
  const status = $(".demo-loading", container);
  const setStatus = (msg) => { status.textContent = msg; };

  setStatus("Loading handovers…");
  const [trajRaw, embRaw] = await Promise.all([loadF32("data/handovers.f32"), loadF32("data/embeddings.f32")]);
  const trajectories = Array.from({ length: N_TRAJ }, (_, i) => trajRaw.subarray(i * T * DOF, (i + 1) * T * DOF));
  const features = standardise(Array.from({ length: N_TRAJ }, (_, i) => Array.from(embRaw.subarray(i * FEAT, (i + 1) * FEAT))));

  // --- renderer / scene -----------------------------------------------------
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  // The scene is limited by pixels drawn, not geometry, so the pixel ratio is the
  // main cost. Start at up to 1.5x and drop to 1x if frames run long (see frame()).
  const DPR_STEPS = [Math.min(window.devicePixelRatio, 1.5), Math.min(window.devicePixelRatio, 1)];
  let dprStep = 0;
  renderer.setPixelRatio(DPR_STEPS[dprStep]);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  viewport.appendChild(renderer.domElement);
  const labelRenderer = new CSS2DRenderer();
  labelRenderer.domElement.className = "demo-labels";
  viewport.appendChild(labelRenderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf3f1ec);
  scene.fog = new THREE.Fog(0xf3f1ec, 6, 14);
  const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 50);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minDistance = 0.6;
  controls.maxDistance = 7;

  scene.add(new THREE.HemisphereLight(0xffffff, 0xb9ab98, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  sun.position.set(2.5, -1.5, 4);
  scene.add(sun);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(30, 30), new THREE.MeshLambertMaterial({ color: 0xe9e5dd }));
  floor.position.z = -0.74;
  scene.add(floor);
  const grid = new THREE.GridHelper(12, 48, 0xd6d0c4, 0xdfdad0);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -0.739;
  scene.add(grid);

  setStatus("Loading the Jaco2 model…");
  const materials = makeMaterials();
  const baseRobot = await loadRobot(materials);
  // The gripper stays closed around the cup, so everything between two moving arm
  // joints is rigid: merge each such body into one mesh per material (21 -> ~12 meshes).
  closeGripper(baseRobot);
  baseRobot.links[EE_LINK].add(makeCup());
  const armJoints = new Set(ARM_JOINTS.map((name) => baseRobot.joints[name]));
  for (const body of [baseRobot, ...armJoints]) mergeStatic(body, armJoints);
  const helper = baseRobot.clone(); // used for forward kinematics only

  const lanes = LANES.map((spec, i) => {
    const group = new THREE.Group();
    group.position.y = (i - 1) * LANE_SPACING;
    scene.add(group);
    // Robot, table and hand paths form a rig turned 90° clockwise (seen from above)
    // about the robot base; the person stays where they are.
    const rig = new THREE.Group();
    rig.rotation.z = -Math.PI / 2;
    rig.position.x = RIG_OFFSET;
    group.add(rig);
    makeTable(rig);
    makePerson(group);
    const plate = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.11, 0.012, 40), new THREE.MeshLambertMaterial({ color: spec.color }));
    plate.rotation.x = Math.PI / 2;
    plate.position.z = 0.006;
    rig.add(plate);
    const robot = baseRobot.clone();
    rig.add(robot);
    robot.traverse((c) => { if (c.isMesh) c.userData.lane = i; });
    const el = document.createElement("div");
    el.className = "lane-label";
    el.style.setProperty("--lane", spec.color);
    const label = new CSS2DObject(el);
    label.position.set(RIG_OFFSET, 0, 0.82);
    group.add(label);
    return { ...spec, i, group, rig, robot, plate, el, trail: null, index: -1 };
  });

  // --- state ------------------------------------------------------------------
  const ui = {
    round: $("[data-round]", container),
    cards: [...container.querySelectorAll("[data-card]")],
    submit: $("[data-submit]", container),
    clear: $("[data-clear]", container),
    best: $("[data-best]", container),
    hint: $("[data-hint]", container),
    algos: [...container.querySelectorAll("[data-algo]")],
    caption: $("[data-caption]", container),
    restart: $("[data-restart]", container),
    speed: $("[data-speed]", container),
    paths: $("[data-paths]", container),
    replay: $("[data-replay]", container),
    views: [...container.querySelectorAll("[data-view]")],
    thinking: $("[data-thinking]", container),
  };
  const state = {
    method: "CMA-ES-IG",
    rng: null, belief: null, algorithm: null,
    query: [], order: [], history: [], round: 0,
    speed: parseFloat(ui.speed.value), restartAt: performance.now(), mode: "rank", busy: false,
  };

  function reset() {
    const seed = (Math.random() * 2 ** 32) >>> 0;
    state.rng = new RNG(seed);
    state.belief = new ParticleBelief(FEAT, new RNG(seed ^ 0x9e3779b9));
    state.algorithm = new ALGORITHMS[state.method](features, { k: K, rng: new RNG(seed ^ 0x85ebca6b) });
    state.round = 0;
    state.history = [];
    state.order = [];
    state.mode = "rank";
    state.query = state.algorithm.query(state.belief);
    showQuery();
  }

  // CMA-ES methods: the handover nearest the search-distribution mean, which is
  // where the candidates are drawn from. Infogain has no search distribution, so
  // it uses the handover with the highest expected reward under the belief.
  const bestIndex = () => {
    const dist = state.algorithm.searchDistribution();
    if (dist) {
      let best = 0, bd = Infinity;
      features.forEach((f, i) => { const d = f.reduce((s, x, j) => s + (x - dist.mean[j]) ** 2, 0); if (d < bd) { bd = d; best = i; } });
      return best;
    }
    const w = state.belief.mean();
    let best = 0, bv = -Infinity;
    features.forEach((f, i) => { const v = f.reduce((s, x, j) => s + x * w[j], 0); if (v > bv) { bv = v; best = i; } });
    return best;
  };

  // --- trajectories in the scene --------------------------------------------
  const q = new Float64Array(DOF);
  const tmp = new THREE.Vector3();
  const quat = new THREE.Quaternion();

  // Orientation markers along the hand path: arrows along the cup's +Y axis in
  // the end-effector frame (bottom to open top), so you can see how the cup tilts.
  const N_MARKERS = 8;
  const shaftGeom = new THREE.CylinderGeometry(0.003, 0.003, 0.04, 6).translate(0, 0.02, 0);
  const coneGeom = new THREE.ConeGeometry(0.011, 0.025, 12).translate(0, 0.0525, 0);

  function setTrajectory(lane, index, color) {
    lane.index = index;
    const traj = trajectories[index];
    const pts = [], poses = [];
    for (let s = 0; s <= 60; s++) {
      setArm(helper, sampleTraj(traj, s / 60, q));
      helper.updateMatrixWorld(true);
      helper.links[EE_LINK].getWorldPosition(tmp);
      pts.push(tmp.clone());
      if (s % Math.round(60 / (N_MARKERS - 1)) === 0 || s === 60) {
        helper.links[EE_LINK].getWorldQuaternion(quat);
        poses.push([tmp.clone(), quat.clone()]);
      }
    }
    if (lane.trail) {
      lane.rig.remove(lane.trail);
      lane.trail.children[0].geometry.dispose();
      lane.trail.children[0].material.dispose();
    }
    const mat = new THREE.MeshLambertMaterial({ color, transparent: true, opacity: 0.85 });
    lane.trail = new THREE.Group();
    lane.trail.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 120, 0.006, 8, false), mat));
    for (const [p, r] of poses) {
      const marker = new THREE.Group();
      marker.position.copy(p);
      marker.quaternion.copy(r);
      marker.add(new THREE.Mesh(shaftGeom, mat), new THREE.Mesh(coneGeom, mat));
      lane.trail.add(marker);
    }
    mergeStatic(lane.trail); // tube + arrows -> one mesh (children[0], disposed above on the next query)
    lane.trail.visible = ui.paths.checked;
    lane.rig.add(lane.trail);
  }

  function showQuery() {
    lanes.forEach((lane, i) => setTrajectory(lane, state.query[i], lane.color));
    state.restartAt = performance.now();
    refresh();
  }

  // --- UI ---------------------------------------------------------------------
  function refresh() {
    const ranking = state.mode === "rank";
    ui.round.textContent = state.mode === "best" ? "Robot's best guess" : `Round ${state.round + 1}`;
    ui.caption.textContent = CAPTIONS[state.method];
    ui.algos.forEach((b) => {
      const on = b.dataset.algo === state.method;
      b.classList.toggle("active", on);
      b.setAttribute("aria-checked", on);
      b.disabled = state.busy;
    });
    lanes.forEach((lane) => {
      const place = state.order.indexOf(lane.i);
      const rankText = place >= 0 ? PLACES[place] : "";
      const card = ui.cards[lane.i];
      card.classList.toggle("ranked", place >= 0);
      card.querySelector("[data-place]").textContent = rankText || "—";
      card.disabled = !ranking;
      lane.el.innerHTML = `<b>${lane.name}</b>${rankText ? `<span>${rankText}</span>` : ""}`;
      const visible = ranking || lane.i === 1;
      lane.group.visible = visible;
    });
    if (!ranking) lanes[1].el.innerHTML = `<b>★</b><span>best guess</span>`;
    lanes[1].plate.material.color.set(ranking ? lanes[1].color : BEST_COLOR);
    ui.submit.disabled = !ranking || state.order.length < K || state.busy;
    ui.clear.disabled = !ranking || !state.order.length;
    ui.best.disabled = state.round === 0 || state.busy;
    ui.best.textContent = ranking ? "Show robot's best guess" : "Back to ranking";
    ui.hint.textContent = !ranking
      ? "This is the handover the robot currently thinks you like most."
      : state.order.length === 0
        ? (state.round === 0 ? "Watch all three, then click your favourite first." : "New candidates! Click your favourite first.")
        : state.order.length < K ? "Now your second favourite." : "Happy with that? Submit your ranking.";
  }

  function pick(laneIdx) {
    if (state.mode !== "rank" || state.busy) return;
    if (state.order.includes(laneIdx)) {
      state.order = state.order.slice(0, state.order.indexOf(laneIdx)); // un-rank it and anything after
    } else {
      state.order.push(laneIdx);
      if (state.order.length === K - 1) state.order.push([0, 1, 2].find((p) => !state.order.includes(p)));
    }
    refresh();
  }

  ui.cards.forEach((card, i) => card.addEventListener("click", () => pick(i)));
  ui.clear.addEventListener("click", () => { state.order = []; refresh(); });
  ui.submit.addEventListener("click", () => {
    if (state.order.length < K || state.busy) return;
    state.busy = true;
    ui.thinking.hidden = false;
    refresh();
    setTimeout(() => { // let the "thinking" indicator paint first
      const ranked = state.order.map((p) => state.query[p]);
      state.belief.update(ranked.map((i) => features[i]));
      state.algorithm.tell(state.query, state.order);
      state.history.push(ranked);
      state.round += 1;
      state.order = [];
      state.query = state.algorithm.query(state.belief);
      state.busy = false;
      ui.thinking.hidden = true;
      showQuery();
      if (state.round === 10) toast("10 rankings — as many as each participant gave in our study. Peek at the robot's best guess!");
    }, 30);
  });
  function setMode(mode) {
    state.mode = mode;
    if (mode === "best") setTrajectory(lanes[1], bestIndex(), BEST_COLOR);
    else setTrajectory(lanes[1], state.query[1], lanes[1].color);
    state.restartAt = performance.now();
    refresh();
  }
  ui.best.addEventListener("click", () => {
    if (state.mode === "rank") { setMode("best"); setView("B"); }
    else { setMode("rank"); setView("All"); }
  });
  ui.algos.forEach((b) => b.addEventListener("click", () => {
    if (state.busy || b.dataset.algo === state.method) return;
    state.method = b.dataset.algo;
    reset();
    setView("All");
  }));
  ui.restart.addEventListener("click", () => { reset(); setView("All"); });
  ui.replay.addEventListener("click", () => { state.restartAt = performance.now(); });
  ui.speed.addEventListener("input", () => {
    const now = performance.now();
    const old = (REAL_DURATION / state.speed + HOLD) * 1000;
    const frac = ((now - state.restartAt) % old) / old;
    state.speed = parseFloat(ui.speed.value);
    state.restartAt = now - frac * (REAL_DURATION / state.speed + HOLD) * 1000;
    $("[data-speed-value]", container).textContent = `${state.speed}×`;
  });
  ui.paths.addEventListener("change", () => lanes.forEach((l) => { if (l.trail) l.trail.visible = ui.paths.checked; }));
  ui.views.forEach((b) => b.addEventListener("click", () => {
    if (state.mode === "best") setMode("rank"); // camera buttons always return to the three candidates
    setView(b.dataset.view);
  }));

  function toast(msg) {
    const t = $(".demo-toast", container);
    t.textContent = msg;
    t.hidden = false;
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => { t.hidden = true; }, 7000);
  }

  // click a robot in the viewport to rank it
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let downAt = null;
  renderer.domElement.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return; // was a drag
    const r = renderer.domElement.getBoundingClientRect();
    pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(lanes.map((l) => l.robot), true).find((h) => h.object.userData.lane !== undefined);
    if (hit) pick(hit.object.userData.lane);
  });

  // --- camera -----------------------------------------------------------------
  const camGoal = { pos: new THREE.Vector3(), target: new THREE.Vector3(), t: 1 };
  const camFrom = { pos: new THREE.Vector3(), target: new THREE.Vector3() };
  let currentView = "All";
  function setView(which, instant = false) {
    currentView = which;
    if (which === "All") {
      // fit all three lanes horizontally and the table-to-label height vertically
      const halfV = THREE.MathUtils.degToRad(camera.fov / 2);
      const halfH = Math.atan(Math.tan(halfV) * camera.aspect);
      const dist = 1.08 * Math.max((LANE_SPACING + 0.75) / Math.tan(halfH), 0.95 / Math.tan(halfV));
      const dir = new THREE.Vector3(1, -0.38, 0.5).normalize();
      camGoal.target.set(0.5, -0.3, 0.12);
      camGoal.pos.copy(camGoal.target).addScaledVector(dir, dist);
    } else {
      const y = (["A", "B", "C"].indexOf(which) - 1) * LANE_SPACING;
      const far = state.mode === "best" ? 1.3 : 1;
      // side-on to the robot-to-person line, so both the cup and the person are visible
      camGoal.target.set(0.65, y - 0.3, 0.2);
      camGoal.pos.set(0.65 + 0.85 * far, y - 0.3 + 1.15 * far, 0.2 + 0.8 * far);
    }
    camFrom.pos.copy(camera.position);
    camFrom.target.copy(controls.target);
    camGoal.t = instant ? 1 : 0;
    if (instant) { camera.position.copy(camGoal.pos); controls.target.copy(camGoal.target); }
    ui.views.forEach((b) => b.classList.toggle("active", b.dataset.view === which));
  }

  function resize() {
    const w = viewport.clientWidth, h = viewport.clientHeight;
    renderer.setSize(w, h);
    labelRenderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    if (currentView === "All") setView("All", true);
  }
  new ResizeObserver(resize).observe(viewport);
  resize();

  // --- animation loop -----------------------------------------------------------
  // Frame times are collected in windows; if more than 10% of the frames in a
  // window took longer than 20 ms (i.e. missed a 60 Hz vsync), step the pixel
  // ratio down. The first window after (re)starting is skipped because it
  // includes shader compilation.
  const FRAME_WINDOW = 90, SLOW_FRAME_MS = 20;
  let frameTimes = [], lastFrame = null, warmup = true;
  function trackFrameTime(now) {
    if (lastFrame !== null && !state.busy) frameTimes.push(now - lastFrame);
    lastFrame = now;
    if (frameTimes.length < FRAME_WINDOW) return;
    const slowShare = frameTimes.filter((t) => t > SLOW_FRAME_MS).length / frameTimes.length;
    frameTimes = [];
    if (warmup) { warmup = false; return; }
    if (slowShare > 0.1 && dprStep < DPR_STEPS.length - 1 && DPR_STEPS[dprStep + 1] < DPR_STEPS[dprStep]) {
      renderer.setPixelRatio(DPR_STEPS[++dprStep]);
      resize();
    }
  }

  // Only render while the demo is on screen and the tab is visible.
  let onScreen = true, running = false;
  function start() {
    if (running || !onScreen || document.hidden) return;
    running = true;
    lastFrame = null;
    frameTimes = [];
    warmup = true;
    requestAnimationFrame(frame);
  }
  new IntersectionObserver(([entry]) => { onScreen = entry.isIntersecting; start(); }).observe(viewport);
  document.addEventListener("visibilitychange", start);

  function frame(now) {
    if (!onScreen || document.hidden) { running = false; return; }
    trackFrameTime(now);
    const duration = (REAL_DURATION / state.speed) * 1000;
    const phase = Math.min(((now - state.restartAt) % (duration + HOLD * 1000)) / duration, 1);
    for (const lane of lanes) {
      if (!lane.group.visible || lane.index < 0) continue;
      setArm(lane.robot, sampleTraj(trajectories[lane.index], retime(phase), q));
    }
    if (camGoal.t < 1) {
      camGoal.t = Math.min(1, camGoal.t + 0.035);
      const e = 1 - (1 - camGoal.t) ** 3;
      camera.position.lerpVectors(camFrom.pos, camGoal.pos, e);
      controls.target.lerpVectors(camFrom.target, camGoal.target, e);
    }
    controls.update();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
    if (debug) debug(now);
    requestAnimationFrame(frame);
  }

  // Add ?debug to the URL for a performance readout (fps, pixel ratio, draw calls, GPU).
  const debug = new URLSearchParams(location.search).has("debug") && (() => {
    const el = document.createElement("pre");
    el.style.cssText = "position:absolute;right:8px;bottom:8px;z-index:5;margin:0;padding:6px 8px;font:11px/1.4 ui-monospace,monospace;background:rgba(0,0,0,.72);color:#fff;border-radius:6px;pointer-events:none";
    viewport.appendChild(el);
    const gl = renderer.getContext();
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const gpu = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    window.__demo = { renderer, scene, lanes };
    let frames = 0, since = performance.now();
    return (now) => {
      frames++;
      if (now - since < 500) return;
      const { calls, triangles } = renderer.info.render;
      el.textContent = `${(1000 * frames / (now - since)).toFixed(0)} fps · ${renderer.getPixelRatio()}x\n${calls} draw calls · ${(triangles / 1000).toFixed(0)}k tris\n${gpu}`;
      frames = 0; since = now;
    };
  })();

  reset();
  setView("All", true);
  status.hidden = true;
  container.classList.add("ready");
  start();
}

const container = document.querySelector("#demo");
if (container) {
  initDemo(container).catch((err) => {
    console.error(err);
    const s = container.querySelector(".demo-loading");
    s.hidden = false;
    s.textContent = `Sorry — the demo failed to load (${err.message}).`;
  });
}
