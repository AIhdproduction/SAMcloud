"""Canvas based point cloud preview used by the sparse and dense pages."""

from __future__ import annotations

from html import escape
import json

from nicegui import ui


_STYLE = """
<style>
.sam-cloud-viewer { position:relative; height:calc(100vh - 180px); min-height:620px; overflow:hidden; border:1px solid #28475b; border-radius:9px; background:#050b12; }
.sam-cloud-viewer:fullscreen { width:100vw; height:100vh; min-height:0; border:0; border-radius:0; }
.sam-cloud-viewer canvas { display:block; width:100%; height:100%; cursor:grab; touch-action:none; }
.sam-cloud-viewer canvas.is-dragging { cursor:grabbing; }
.sam-cloud-toolbar { position:absolute; top:12px; left:12px; right:12px; display:flex; flex-wrap:wrap; gap:6px; align-items:center; pointer-events:none; }
.sam-cloud-toolbar > * { pointer-events:auto; }.sam-cloud-toolbar button { min-height:30px; padding:3px 9px; border:1px solid #47718a; border-radius:5px; background:#102131; color:var(--sam-text,#eaf2f8); font:13px inherit; cursor:pointer; }.sam-cloud-toolbar button:hover { border-color:var(--sam-accent,#26b7ef); }.sam-cloud-toolbar button.is-selected { border-color:var(--sam-accent,#26b7ef); background:#16516c; }
.sam-cloud-toolbar label { display:flex; align-items:center; gap:7px; padding:3px 8px; border:1px solid #47718a; border-radius:5px; background:#102131; color:var(--sam-text,#eaf2f8); font-size:13px; }.sam-cloud-toolbar input { width:76px; accent-color:var(--sam-accent,#26b7ef); }
.sam-cloud-status { position:absolute; bottom:12px; left:12px; max-width:calc(100% - 24px); padding:7px 10px; border-radius:5px; background:rgba(9,19,29,.86); color:var(--sam-muted,#96aabe); font-size:13px; pointer-events:none; }
.sam-cloud-class-panel { position:absolute; top:62px; right:12px; bottom:12px; display:none; width:min(340px,34vw); overflow:auto; padding:12px; border:1px solid #28475b; border-radius:7px; background:rgba(9,19,29,.94); color:var(--sam-text,#eaf2f8); }.sam-cloud-viewer:fullscreen .sam-cloud-class-panel { display:block; }.sam-cloud-class-panel h3 { margin:0 0 10px; font-size:15px; }.sam-cloud-class-actions { display:flex; gap:6px; margin-bottom:8px; }.sam-cloud-class-panel button { padding:3px 7px; border:1px solid #47718a; border-radius:4px; background:#102131; color:var(--sam-text,#eaf2f8); font:12px inherit; cursor:pointer; }.sam-cloud-class-row { display:grid; grid-template-columns:auto 1fr auto; gap:7px; align-items:center; padding:5px 0; border-top:1px solid rgba(113,154,181,.18); font-size:13px; }
@media (max-width:720px) { .sam-cloud-viewer { height:calc(100vh - 140px); min-height:460px; }.sam-cloud-toolbar { right:auto; max-width:calc(100% - 24px); }.sam-cloud-toolbar label { display:none; } .sam-cloud-class-panel { width:calc(100vw - 24px); } }
</style>
"""


def render_pointcloud_viewer(
    kind: str,
    *,
    title: str,
    class_colors: list[list[int]] | None = None,
    class_labels: list[str] | None = None,
) -> None:
    """Render an interactive, dependency-free 2D projection of a point cloud."""
    if kind not in {"sparse", "dense"}:
        raise ValueError("Point cloud kind must be 'sparse' or 'dense'")
    element_id = f"sam-cloud-{kind}"
    labels = class_labels or []
    colors = class_colors or []
    panel = ""
    if labels:
        rows = "".join(
            f'<div class="sam-cloud-class-row"><input type="checkbox" checked data-class-visible="{index}">'
            f"<span>{escape(label)}</span><button data-class-isolate=\"{index}\">Isolate</button></div>"
            for index, label in enumerate(labels)
        )
        panel = (
            '<aside class="sam-cloud-class-panel"><h3>Class visibility</h3>'
            '<div class="sam-cloud-class-actions"><button data-action="show-all">Show all</button>'
            '<button data-action="hide-all">Hide all</button></div>'
            f"{rows}</aside>"
        )
    ui.add_head_html(_STYLE)
    ui.html(
        f'''<div id="{element_id}" class="sam-cloud-viewer">
<canvas aria-label="{escape(title)}"></canvas>
<div class="sam-cloud-toolbar">
<button data-action="reset">Reset view</button><button data-action="zoom-out">-</button><button data-action="zoom-in">+</button><button data-action="fit">Fit all</button><button data-action="fullscreen">Fullscreen</button>
<button data-view="terrain" class="is-selected">Terrain view</button><button data-view="orbit">Orbit view</button>
<button data-mode="rgb" class="is-selected">RGB</button><button data-mode="classification">Classification</button>
<label>Point size <input data-action="point-size" type="range" min="1" max="6" step="0.5" value="1"></label>
</div>{panel}<div class="sam-cloud-status">Loading point cloud preview...</div></div>''',
        sanitize=False,
    )
    config = json.dumps({
        "elementId": element_id,
        "endpoint": f"/api/pointcloud/{kind}",
        "classColors": colors,
        "classCount": len(labels),
    })
    ui.add_body_html(
        f"<script>(() => {{ const samcloudViewerConfig = {config};\n{_SCRIPT}\n}})();</script>"
    )


_SCRIPT = r"""
(() => {
  const config = samcloudViewerConfig;
  const boot = () => {
    const root = document.getElementById(config.elementId);
    if (!root) return false;
    if (root.dataset.ready === '1') return true;
    root.dataset.ready = '1';
    const canvas = root.querySelector('canvas'), context = canvas.getContext('2d'), status = root.querySelector('.sam-cloud-status');
    const state = { points:null, colors:null, classes:null, baseCenter:[0,0,0], center:[0,0,0], span:1, zoom:1, panX:0, panY:0, yaw:0.65, pitch:-0.45, terrainTilt:0.78, view:'terrain', mode:'rgb', pointSize:1, classColors:config.classColors.slice(), classVisible:Array(Math.max(config.classCount, 1)).fill(true), dragging:null };
    const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
    const setStatus = text => { status.textContent = text; };
    const reset = angles => { state.zoom=1; state.panX=state.panY=0; state.center=state.baseCenter.slice(); if (angles) { state.yaw=0.65; state.pitch=-0.45; state.terrainTilt=0.78; } draw(); };
    const resize = () => { const rect=canvas.getBoundingClientRect(), ratio=window.devicePixelRatio||1; canvas.width=Math.max(1,Math.round(rect.width*ratio)); canvas.height=Math.max(1,Math.round(rect.height*ratio)); context.setTransform(ratio,0,0,ratio,0,0); draw(); };
    const project = (index, width, height) => {
      const offset=index*3, x=state.points[offset]-state.center[0], y=state.points[offset+1]-state.center[1], z=state.points[offset+2]-state.center[2];
      let x1,y1,depth;
      if (state.view === 'terrain') { const sy=Math.sin(state.yaw), cy=Math.cos(state.yaw), st=Math.sin(state.terrainTilt), ct=Math.cos(state.terrainTilt); const groundY=x*sy+y*cy; x1=x*cy-y*sy; y1=groundY*ct+z*st; depth=z*ct-groundY*st; }
      else { const sy=Math.sin(state.yaw), cy=Math.cos(state.yaw), sp=Math.sin(state.pitch), cp=Math.cos(state.pitch); const z1=x*sy+z*cy; x1=x*cy-z*sy; y1=y*cp-z1*sp; depth=y*sp+z1*cp; }
      const depthFactor=clamp(1-depth/(state.span*7),.72,1.35), scale=Math.min(width,height)*.82*state.zoom/state.span;
      return [width/2+state.panX+x1*scale*depthFactor, height/2+state.panY-y1*scale*depthFactor, depthFactor];
    };
    const draw = () => {
      const rect=canvas.getBoundingClientRect(), width=rect.width, height=rect.height;
      context.clearRect(0,0,width,height); context.fillStyle='#050b12'; context.fillRect(0,0,width,height);
      if (!state.points) return;
      const count=state.classes.length, stride=state.dragging ? Math.max(1, Math.ceil(count/120000)) : 1;
      for (let index=0; index<count; index+=stride) {
        const classId=state.classes[index]; if (state.classVisible[classId] === false) continue;
        const item=project(index,width,height), px=item[0], py=item[1], depthFactor=item[2]; if (px < -8 || py < -8 || px > width+8 || py > height+8) continue;
        const offset=index*3;
        if (state.mode === 'classification' && state.classColors.length) { const color=state.classColors[classId % state.classColors.length]; context.fillStyle=`rgb(${color[0]},${color[1]},${color[2]})`; }
        else context.fillStyle=`rgb(${state.colors[offset]},${state.colors[offset+1]},${state.colors[offset+2]})`;
        const size=Math.max(.7,state.pointSize*clamp(depthFactor*1.8,.8,2.5)); context.fillRect(px,py,size,size);
      }
    };
    const updateButtons = () => { root.querySelectorAll('[data-view]').forEach(button => button.classList.toggle('is-selected', button.dataset.view === state.view)); root.querySelectorAll('[data-mode]').forEach(button => button.classList.toggle('is-selected', button.dataset.mode === state.mode)); };
    const closestPoint = event => { if (!state.points) return; const rect=canvas.getBoundingClientRect(), targetX=event.clientX-rect.left, targetY=event.clientY-rect.top; let closest=-1, distance=Infinity; for(let i=0;i<state.classes.length;i++){const item=project(i,rect.width,rect.height), dx=item[0]-targetX, dy=item[1]-targetY, next=dx*dx+dy*dy;if(next<distance){distance=next;closest=i;}} if(closest>=0 && distance<2500){const offset=closest*3;state.center=[state.points[offset],state.points[offset+1],state.points[offset+2]];state.panX=targetX-rect.width/2;state.panY=targetY-rect.height/2;} };
    canvas.addEventListener('pointerdown', event => { if (!state.points) return; event.preventDefault(); state.dragging={button:event.button,x:event.clientX,y:event.clientY}; if(event.button===0) closestPoint(event); canvas.setPointerCapture(event.pointerId);canvas.classList.add('is-dragging');draw(); });
    canvas.addEventListener('pointermove', event => { if (!state.dragging) return; const dx=event.clientX-state.dragging.x, dy=event.clientY-state.dragging.y; state.dragging.x=event.clientX;state.dragging.y=event.clientY; if(state.dragging.button===0){state.yaw+=dx*.008;if(state.view==='terrain')state.terrainTilt=clamp(state.terrainTilt-dy*.008,0,1.25);else state.pitch=clamp(state.pitch-dy*.008,-1.5,1.5);}else{state.panX+=dx;state.panY+=dy;}draw(); });
    const stopDrag = () => { if(state.dragging){state.dragging=null;canvas.classList.remove('is-dragging');draw();} }; canvas.addEventListener('pointerup',stopDrag);canvas.addEventListener('pointercancel',stopDrag);canvas.addEventListener('contextmenu', event=>event.preventDefault());
    canvas.addEventListener('wheel', event=>{event.preventDefault();state.zoom=clamp(state.zoom*(event.deltaY<0?1.15:.87),.25,12);draw();},{passive:false});
    root.addEventListener('click', event => { const button=event.target.closest('button');if(!button)return; const action=button.dataset.action;if(action==='reset')reset(true);if(action==='fit')reset(false);if(action==='zoom-in'||action==='zoom-out'){state.zoom=clamp(state.zoom*(action==='zoom-in'?1.25:.8),.25,12);draw();}if(action==='fullscreen')root.requestFullscreen?.();if(action==='show-all'||action==='hide-all'){state.classVisible.fill(action==='show-all');root.querySelectorAll('[data-class-visible]').forEach(input=>input.checked=action==='show-all');draw();}if(button.dataset.classIsolate!==undefined){const only=Number(button.dataset.classIsolate);state.classVisible=state.classVisible.map((_,index)=>index===only);root.querySelectorAll('[data-class-visible]').forEach(input=>input.checked=Number(input.dataset.classVisible)===only);draw();}if(button.dataset.view){state.view=button.dataset.view;updateButtons();draw();}if(button.dataset.mode){state.mode=button.dataset.mode;updateButtons();draw();} });
    root.addEventListener('input', event => { if(event.target.dataset.action==='point-size'){state.pointSize=Number(event.target.value);draw();}if(event.target.dataset.classVisible!==undefined){state.classVisible[Number(event.target.dataset.classVisible)]=event.target.checked;draw();} });
    window.samcloudPointCloudViewers=window.samcloudPointCloudViewers||{}; window.samcloudPointCloudViewers[config.elementId]={ setClassColors(colors){if(Array.isArray(colors)){state.classColors=colors;draw();}}, setClassVisibility(visible){if(Array.isArray(visible)){state.classVisible=visible.map(Boolean);root.querySelectorAll('[data-class-visible]').forEach(input=>input.checked=state.classVisible[Number(input.dataset.classVisible)]!==false);draw();}} };
    new ResizeObserver(resize).observe(root); resize();
    fetch(config.endpoint,{cache:'no-store'}).then(async response=>{if(!response.ok){let body={};try{body=await response.json();}catch(_){ }throw new Error(body.error||response.statusText);}return response.arrayBuffer();}).then(buffer=>{const view=new DataView(buffer);if(buffer.byteLength<16||view.getUint8(0)!==83||view.getUint8(1)!==65||view.getUint8(2)!==77||view.getUint8(3)!==86)throw new Error('Invalid preview format');const count=view.getUint32(8,true);if(buffer.byteLength<16+count*16)throw new Error('Preview data is incomplete');state.points=new Float32Array(count*3);state.colors=new Uint8Array(count*3);state.classes=new Uint8Array(count);let minimum=[Infinity,Infinity,Infinity],maximum=[-Infinity,-Infinity,-Infinity];for(let i=0,offset=16;i<count;i++,offset+=16){for(let axis=0;axis<3;axis++){const value=view.getFloat32(offset+axis*4,true);state.points[i*3+axis]=value;minimum[axis]=Math.min(minimum[axis],value);maximum[axis]=Math.max(maximum[axis],value);}state.colors[i*3]=view.getUint8(offset+12);state.colors[i*3+1]=view.getUint8(offset+13);state.colors[i*3+2]=view.getUint8(offset+14);state.classes[i]=view.getUint8(offset+15);}state.baseCenter=minimum.map((value,index)=>(value+maximum[index])/2);state.center=state.baseCenter.slice();state.span=Math.max(.001,Math.hypot(maximum[0]-minimum[0],maximum[1]-minimum[1],maximum[2]-minimum[2]));setStatus(`Preview: ${count.toLocaleString()} points. Left drag rotates around the nearest point, right drag moves, wheel zooms.`);draw();}).catch(error=>setStatus(`Point cloud unavailable: ${error.message}`));
    return true;
  };
  if (!boot()) { const observer=new MutationObserver(()=>{if(boot())observer.disconnect();}); observer.observe(document.documentElement,{childList:true,subtree:true}); setTimeout(()=>observer.disconnect(),10000); }
})();
"""
