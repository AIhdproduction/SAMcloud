# Vendored three.js

Version: three@0.170.0 (MIT, see LICENSE), fetched from https://cdn.jsdelivr.net/npm/three@0.170.0/

Files:
- three.module.js  (build/three.module.js, unchanged)
- OrbitControls.js (examples/jsm/controls/OrbitControls.js)
- GLTFLoader.js    (examples/jsm/loaders/GLTFLoader.js)
- BufferGeometryUtils.js (examples/jsm/utils/BufferGeometryUtils.js)

Modification: the bare import specifier `from 'three'` in the three jsm files was rewritten to the
relative `from './three.module.js'`, and `'../utils/BufferGeometryUtils.js'` in GLTFLoader.js to
`'./BufferGeometryUtils.js'`, so the files work offline without an import map from the flat folder
served at /assets-static/vendor/three/. Nothing else was changed.
