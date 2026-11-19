// IFC 3D viewer for SAMcloud: glTF (one node per IFC element, node.name = GlobalId) rendered with three.js.
// three.js is vendored under /assets-static/vendor/three/ (r170) and loaded lazily with dynamic import().
// Visibility is per "category" (a key such as walls/slabs...) that Python supplies through classByNode
// (GlobalId -> category). Nodes missing from the map fall back to node.extras.ifc_class.
export default {
  template: `
    <div style="position:relative;width:100%;height:100%;min-height:420px;overflow:hidden;border:1px solid #28475b;border-radius:9px">
      <div ref="host" style="position:absolute;inset:0;touch-action:none;outline:none"></div>
      <div ref="hint" style="position:absolute;right:10px;bottom:8px;padding:4px 8px;border-radius:5px;background:rgba(9,19,29,.86);color:#7f97ab;font-size:11px;pointer-events:none">{{ (strings && strings.hint) || 'Drag: orbit · right drag: pan · wheel: zoom · click: properties' }}</div>
      <div v-if="message" style="position:absolute;inset:0;display:grid;place-items:center;color:#96aabe;font-size:14px;pointer-events:none;text-align:center;padding:12px">{{ message }}</div>
    </div>
  `,
  props: {
    src: String,
    hiddenClasses: Array,
    classByNode: Object,
    clipHeight: Number,
    background: Array,
    selected: String,
    strings: Object,
  },
  data() {
    return { message: "" };
  },
  async mounted() {
    this.disposed = false;
    this.ready = false;
    this.elements = [];
    this.pickedElement = null;
    this.frameRequested = false;
    this.loadToken = 0;
    try {
      await this.loadThree();
    } catch (error) {
      this.message = ((this.strings && this.strings.load_failed) || "Could not load the 3D model ({error}).").replace("{error}", error.message);
      this.$emit("error", { message: String(error.message || error) });
      return;
    }
    if (this.disposed) return;
    this.setupScene();
    this.loadModel();
  },
  unmounted() {
    this.disposed = true;
    this.teardown();
  },
  watch: {
    src() {
      this.loadModel();
    },
    hiddenClasses() {
      this.applyVisibility();
    },
    classByNode() {
      this.indexElements();
      this.applyVisibility();
    },
    clipHeight() {
      this.applyClip();
    },
    background() {
      this.applyBackground();
    },
    selected() {
      this.highlight(this.selected || null);
    },
  },
  methods: {
    absoluteUrl(url) {
      return (url.startsWith("/") ? window.path_prefix || "" : "") + url;
    },
    async loadThree() {
      const base = this.absoluteUrl("/assets-static/vendor/three/");
      const [THREE, controls, loader] = await Promise.all([
        import(base + "three.module.js"),
        import(base + "OrbitControls.js"),
        import(base + "GLTFLoader.js"),
      ]);
      this.THREE = THREE;
      this.OrbitControls = controls.OrbitControls;
      this.GLTFLoader = loader.GLTFLoader;
    },
    setupScene() {
      const THREE = this.THREE;
      const host = this.$refs.host;
      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
      this.renderer.setPixelRatio(window.devicePixelRatio || 1);
      this.renderer.localClippingEnabled = true;
      host.appendChild(this.renderer.domElement);
      this.renderer.domElement.style.display = "block";
      this.scene = new THREE.Scene();
      this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
      this.camera.position.set(8, 6, 8);
      this.scene.add(new THREE.HemisphereLight(0xffffff, 0x556070, 1.1));
      const sun = new THREE.DirectionalLight(0xffffff, 1.6);
      sun.position.set(5, 10, 7);
      this.scene.add(sun);
      this.sun = sun;
      this.modelRoot = new THREE.Group();
      this.scene.add(this.modelRoot);
      this.clipPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), 0);
      this.controls = new this.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = false;
      this.controls.addEventListener("change", () => this.requestDraw());
      this.raycaster = new THREE.Raycaster();
      this.pointerDown = null;
      const dom = this.renderer.domElement;
      this.onPointerDown = (event) => {
        this.pointerDown = { x: event.clientX, y: event.clientY, button: event.button };
      };
      this.onPointerUp = (event) => {
        const down = this.pointerDown;
        this.pointerDown = null;
        if (!down || down.button !== 0) return;
        if (Math.hypot(event.clientX - down.x, event.clientY - down.y) > 4) return;
        this.pick(event);
      };
      dom.addEventListener("pointerdown", this.onPointerDown);
      dom.addEventListener("pointerup", this.onPointerUp);
      this.observer = new ResizeObserver(() => this.resize());
      this.observer.observe(this.$el);
      this.applyBackground();
      this.resize();
    },
    teardown() {
      if (this.observer) this.observer.disconnect();
      this.observer = null;
      if (this.renderer) {
        const dom = this.renderer.domElement;
        dom.removeEventListener("pointerdown", this.onPointerDown);
        dom.removeEventListener("pointerup", this.onPointerUp);
      }
      if (this.controls) this.controls.dispose();
      this.controls = null;
      this.disposeModel();
      if (this.renderer) {
        this.renderer.dispose();
        if (this.renderer.forceContextLoss) this.renderer.forceContextLoss();
        if (this.renderer.domElement.parentNode) this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
      }
      this.renderer = null;
      this.scene = null;
    },
    disposeModel() {
      this.elements = [];
      this.pickedElement = null;
      if (!this.modelRoot) return;
      this.modelRoot.traverse((object) => {
        if (object.geometry) object.geometry.dispose();
        if (object.material) {
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          for (const material of materials) {
            for (const value of Object.values(material)) {
              if (value && value.isTexture) value.dispose();
            }
            material.dispose();
          }
        }
      });
      this.modelRoot.clear();
    },
    applyBackground() {
      if (!this.scene) return;
      const [r, g, b] = this.background && this.background.length === 3 ? this.background : [9, 19, 29];
      this.scene.background = new this.THREE.Color(r / 255, g / 255, b / 255);
      this.requestDraw();
    },
    resize() {
      if (!this.renderer) return;
      const rect = this.$el.getBoundingClientRect();
      const width = Math.max(1, Math.round(rect.width));
      const height = Math.max(1, Math.round(rect.height));
      this.renderer.setSize(width, height, false);
      this.renderer.domElement.style.width = width + "px";
      this.renderer.domElement.style.height = height + "px";
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.requestDraw();
    },
    requestDraw() {
      if (this.frameRequested || !this.renderer) return;
      this.frameRequested = true;
      requestAnimationFrame(() => {
        this.frameRequested = false;
        if (!this.renderer || this.disposed) return;
        this.renderer.render(this.scene, this.camera);
      });
    },
    loadModel() {
      if (!this.scene) return;
      this.disposeModel();
      this.ready = false;
      this.requestDraw();
      if (!this.src) {
        this.message = (this.strings && this.strings.no_model) || "No 3D model.";
        return;
      }
      const token = ++this.loadToken;
      this.message = (this.strings && this.strings.loading) || "Loading 3D model...";
      const loader = new this.GLTFLoader();
      loader.load(
        this.absoluteUrl(this.src),
        (gltf) => {
          if (this.disposed || token !== this.loadToken) return;
          this.onLoaded(gltf);
        },
        undefined,
        (error) => {
          if (this.disposed || token !== this.loadToken) return;
          const text = String((error && error.message) || error);
          this.message = ((this.strings && this.strings.load_failed) || "Could not load the 3D model ({error}).").replace("{error}", text);
          this.$emit("error", { message: text });
        },
      );
    },
    onLoaded(gltf) {
      const THREE = this.THREE;
      const scene = gltf.scene;
      // glTF is Y-up. If the exporter left the model Z-up (it is flat in Z, tall in Y) convert it here.
      const box0 = new THREE.Box3().setFromObject(scene);
      const size0 = box0.getSize(new THREE.Vector3());
      this.upConverted = false;
      if (size0.y > size0.z * 1.5 && size0.z < size0.x) {
        scene.rotation.x = -Math.PI / 2;
        this.upConverted = true;
      }
      this.modelRoot.add(scene);
      scene.updateMatrixWorld(true);
      this.prepareMaterials(scene);
      this.indexElements();
      this.applyVisibility();
      this.applyClip();
      this.message = "";
      this.fit();
      this.ready = true;
      const box = this.visibleBox();
      this.$emit("ready", {
        elements: this.elements.length,
        min_height: box ? box.min.y : 0,
        max_height: box ? box.max.y : 0,
        up_converted: this.upConverted,
      });
      if (this.selected) this.highlight(this.selected);
    },
    prepareMaterials(scene) {
      const THREE = this.THREE;
      scene.traverse((object) => {
        if (!object.isMesh) return;
        const source = Array.isArray(object.material) ? object.material : [object.material];
        const cloned = source.map((material) => {
          const copy = material.clone();
          copy.userData = Object.assign({}, material.userData, {
            baseColor: copy.color ? copy.color.getHex() : null,
            baseEmissive: copy.emissive ? copy.emissive.getHex() : null,
            baseSide: copy.side,
            baseOpacity: copy.opacity,
            baseTransparent: copy.transparent,
          });
          copy.clippingPlanes = [];
          copy.clipShadows = false;
          return copy;
        });
        object.material = Array.isArray(object.material) ? cloned : cloned[0];
        if (!object.geometry.attributes.normal) object.geometry.computeVertexNormals();
        if (!object.material.vertexColors && object.geometry.attributes.color && !Array.isArray(object.material)) {
          object.material.vertexColors = true;
        }
      });
    },
    elementClass(root) {
      const extras = root.userData || {};
      return extras.ifc_class || "";
    },
    indexElements() {
      const map = this.classByNode || {};
      const found = [];
      const walk = (object, owner) => {
        let current = owner;
        const guid = object.name;
        const extras = object.userData || {};
        if (!current && (extras.ifc_class || (guid && Object.prototype.hasOwnProperty.call(map, guid)))) {
          current = { root: object, guid, ifcClass: extras.ifc_class || "", name: extras.name || "", category: "", meshes: [] };
          current.category = Object.prototype.hasOwnProperty.call(map, guid) ? map[guid] : current.ifcClass || "other";
          found.push(current);
        }
        if (current && object.isMesh) {
          current.meshes.push(object);
          object.userData.__element = current;
        }
        for (const child of object.children) walk(child, current);
      };
      if (this.modelRoot) for (const child of this.modelRoot.children) walk(child, null);
      for (const element of found) this.applyGlass(element);
      this.elements = found;
    },
    applyGlass(element) {
      const cls = element.ifcClass || element.category;
      const opacity = cls === "IfcSpace" || cls === "spaces" ? 0.22 : cls === "IfcWindow" || cls === "windows" ? 0.35 : 1;
      for (const mesh of element.meshes) {
        const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        for (const material of materials) {
          const base = material.userData.baseOpacity === undefined ? 1 : material.userData.baseOpacity;
          const target = Math.min(base, opacity);
          material.opacity = target;
          material.transparent = target < 1 || !!material.userData.baseTransparent;
          material.depthWrite = target >= 1;
        }
      }
    },
    applyVisibility() {
      const hidden = new Set(this.hiddenClasses || []);
      for (const element of this.elements) {
        element.root.visible = !hidden.has(element.category);
      }
      if (this.pickedElement && !this.pickedElement.root.visible) this.highlight(null, true);
      this.requestDraw();
    },
    applyClip() {
      const active = typeof this.clipHeight === "number" && Number.isFinite(this.clipHeight);
      if (active) this.clipPlane.constant = this.clipHeight;
      if (!this.modelRoot) return;
      this.modelRoot.traverse((object) => {
        if (!object.isMesh) return;
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        for (const material of materials) {
          material.clippingPlanes = active ? [this.clipPlane] : [];
          material.side = active ? this.THREE.DoubleSide : material.userData.baseSide;
          material.needsUpdate = true;
        }
      });
      this.requestDraw();
    },
    visibleBox() {
      const THREE = this.THREE;
      const box = new THREE.Box3();
      let any = false;
      if (this.elements.length === 0) {
        box.setFromObject(this.modelRoot);
        return box.isEmpty() ? null : box;
      }
      for (const element of this.elements) {
        if (!element.root.visible) continue;
        const part = new THREE.Box3().setFromObject(element.root);
        if (part.isEmpty()) continue;
        box.union(part);
        any = true;
      }
      if (!any) {
        box.setFromObject(this.modelRoot);
        return box.isEmpty() ? null : box;
      }
      return box;
    },
    fit() {
      if (!this.camera || !this.modelRoot) return;
      const THREE = this.THREE;
      const box = this.visibleBox();
      if (!box) return;
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const radius = Math.max(size.length() / 2, 0.5);
      const distance = radius / Math.sin((this.camera.fov * Math.PI) / 360);
      const direction = new THREE.Vector3(1, 0.8, 1).normalize();
      this.camera.position.copy(center).addScaledVector(direction, distance);
      this.camera.near = Math.max(distance / 500, 0.01);
      this.camera.far = distance * 20;
      this.camera.updateProjectionMatrix();
      this.controls.target.copy(center);
      this.controls.update();
      this.requestDraw();
    },
    pickables() {
      const meshes = [];
      for (const element of this.elements) {
        if (!element.root.visible) continue;
        for (const mesh of element.meshes) {
          let visible = true;
          for (let node = mesh; node; node = node.parent) {
            if (!node.visible) {
              visible = false;
              break;
            }
          }
          if (visible) meshes.push(mesh);
        }
      }
      return meshes;
    },
    pick(event) {
      if (!this.ready) return;
      const THREE = this.THREE;
      const rect = this.renderer.domElement.getBoundingClientRect();
      const pointer = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );
      this.raycaster.setFromCamera(pointer, this.camera);
      const active = typeof this.clipHeight === "number" && Number.isFinite(this.clipHeight);
      const hits = this.raycaster.intersectObjects(this.pickables(), false).filter((hit) => !active || hit.point.y <= this.clipHeight + 1e-6);
      let chosen = null;
      for (const hit of hits) {
        const element = hit.object.userData.__element;
        if (!element) continue;
        const space = element.ifcClass === "IfcSpace" || element.category === "spaces";
        if (!space) {
          chosen = element;
          break;
        }
        if (!chosen) chosen = element;
      }
      this.highlight(chosen, true);
      this.$emit("pick", { global_id: chosen ? chosen.guid : null, ifc_class: chosen ? chosen.ifcClass : "" });
    },
    highlight(target, byElement) {
      let element = target;
      if (!byElement) element = target ? this.elements.find((item) => item.guid === target) || null : null;
      if (this.pickedElement) this.paint(this.pickedElement, false);
      this.pickedElement = element || null;
      if (this.pickedElement) this.paint(this.pickedElement, true);
      this.requestDraw();
    },
    paint(element, on) {
      for (const mesh of element.meshes) {
        const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        for (const material of materials) {
          if (material.emissive) {
            material.emissive.setHex(on ? 0x2a6bff : material.userData.baseEmissive || 0x000000);
            material.emissiveIntensity = on ? 0.9 : 1;
          } else if (material.color && material.userData.baseColor !== null) {
            material.color.setHex(on ? 0x2a6bff : material.userData.baseColor);
          }
        }
      }
    },
  },
};
