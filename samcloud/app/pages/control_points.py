"""Ground-control-point import and image-marking workspace."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from nicegui import events, ui

from samcloud.app.layout import render_page
from samcloud.app.pages.gcp_review import render_review_workspace
from samcloud.app.state import state
from samcloud.control_points.repository import apply_roles, load_observations, read_control_points, save_observations, write_control_points
from samcloud.control_points.georeference import georeference_from_control_points
from samcloud.engine.pipeline import _create_pristine_gcp_backups


_CRS_OPTIONS = {"EPSG:2056+5728": "LV95 + LN02", "EPSG:2056": "LV95 ellipsoidal", "EPSG:4979": "WGS84 ellipsoidal"}


def render() -> None:
    """Render imported GCPs and clicks on undistorted images."""
    def content() -> None:
        project_directory = state.active_project_directory
        project = state.active_project()
        if project_directory is None or project is None:
            ui.label("Open a project before managing control points.")
            return
        control_directory = project_directory / "control_points"
        csv_path, observations_path = control_directory / "control_points.csv", control_directory / "observations.json"
        settings = project.setdefault("settings", {}).setdefault("control_points", {})
        settings.setdefault("crs", "EPSG:2056+5728"); settings.setdefault("min_points", 3); settings.setdefault("min_observations_per_point", 2)
        selected = {"name": None, "image": None}
        preview = {"report": None, "error": ""}
        pending_import: dict[str, object] = {"points": None}

        def points(): return read_control_points(csv_path) if csv_path.exists() else None
        def clicks(): return load_observations(observations_path)
        def save_crs(event: events.ValueChangeEventArguments) -> None:
            settings["crs"] = event.value; state.projects.save(project_directory, project)

        def save_role(name: str, role: str) -> None:
            control_points = points()
            if control_points is not None:
                write_control_points(apply_roles(control_points, {name: role}), csv_path)
                table.refresh()

        with ui.card().classes("w-full"):
            ui.label("Control point coordinates").classes("text-lg font-medium")
            ui.label("Import name, x, y, z coordinates. The default is Swiss LV95 + LN02.").classes("sam-muted text-sm")
            ui.select(_CRS_OPTIONS, label="Control-point CRS", value=settings["crs"], on_change=save_crs).classes("w-full max-w-md")
            async def import_csv(event: events.UploadEventArguments) -> None:
                upload = control_directory / ".uploaded_control_points"
                await event.file.save(upload)
                try:
                    pending_import["points"] = read_control_points(upload)
                except ValueError as error: ui.notify(str(error), type="negative")
                import_table.refresh()
            ui.upload(on_upload=import_csv, auto_upload=True, label="Punktdatei importieren").props("accept=.csv,.txt,.dat").classes("q-mt-md")

        @ui.refreshable
        def import_table() -> None:
            imported = pending_import["points"]
            if imported is None:
                return
            control_points = imported
            ui.label("Rollen vor dem Import festlegen").classes("text-lg font-medium q-mt-md")
            with ui.row().classes("gap-2 q-mb-sm"):
                def set_all_pass() -> None:
                    current = pending_import["points"]
                    pending_import["points"] = apply_roles(current, {str(name): "pass" for name in current["name"]})
                    import_table.refresh()
                def assign_cp_prefix() -> None:
                    current = pending_import["points"]
                    pending_import["points"] = apply_roles(current, {str(name): "control" for name in current["name"] if str(name).casefold().startswith("cp")})
                    import_table.refresh()
                ui.button("Alle als Passpunkte", on_click=set_all_pass).props("outline")
                ui.button("CP-Präfix automatisch zuordnen", on_click=assign_cp_prefix).props("outline")
            with ui.element("table").classes("w-full"):
                with ui.element("thead"):
                    with ui.element("tr"):
                        for label in ("Name", "X", "Y", "Z", "Rolle"):
                            with ui.element("th"):
                                ui.label(label)
                with ui.element("tbody"):
                    for row in control_points.to_dict("records"):
                        with ui.element("tr"):
                            for key in ("name", "x", "y", "z"):
                                with ui.element("td"):
                                    ui.label(str(row[key]))
                            with ui.element("td"):
                                def set_role(event: events.ValueChangeEventArguments, name: str = str(row["name"])) -> None:
                                    pending_import["points"] = apply_roles(control_points, {name: event.value})
                                ui.select({"pass": "Passpunkt", "control": "Kontrollpunkt"}, value=row["role"], on_change=set_role).classes("w-48")
            def commit_import() -> None:
                write_control_points(pending_import["points"], csv_path)
                pending_import["points"] = None
                import_table.refresh(); table.refresh(); workspace.refresh()
            ui.button("Import", on_click=commit_import).classes("sam-button-primary q-mt-sm")

        @ui.refreshable
        def table() -> None:
            if (control_directory / "detections.json").is_file():
                return
            control_points = points()
            if control_points is None:
                ui.label("No control-point CSV imported yet.").classes("sam-muted q-mt-md"); return
            current = clicks()
            report_path = project_directory / "reports" / "control_points_report.json"
            residuals = {}
            if report_path.exists():
                try: residuals = {item["name"]: item for item in json.loads(report_path.read_text(encoding="utf-8")).get("points", [])}
                except (OSError, ValueError): pass
            rows = [{**row, "observations": len(current.get(str(row["name"]), [])), "status": "Ready" if len(current.get(str(row["name"]), [])) >= int(settings["min_observations_per_point"]) else "Needs two images", "residual_m": residuals.get(str(row["name"]), {}).get("residual_m", ""), "leave_one_out_m": residuals.get(str(row["name"]), {}).get("leave_one_out_residual_m", "")} for row in control_points.to_dict("records")]
            columns = [{"name": key, "label": label, "field": key} for key, label in (("name", "Name"), ("x", "X"), ("y", "Y"), ("z", "Z"), ("role", "Rolle"), ("observations", "Images"), ("status", "Status"), ("residual_m", "Residual m"), ("leave_one_out_m", "Leave-one-out m"))]
            ui.table(columns=columns, rows=rows, row_key="name").classes("w-full q-mt-md")
            ui.label("Rolle vor der Verarbeitung ändern").classes("sam-muted text-sm q-mt-sm")
            for row in control_points.to_dict("records"):
                with ui.row().classes("items-center gap-2"):
                    ui.label(str(row["name"])).classes("w-40")
                    ui.select({"pass": "Passpunkt", "control": "Kontrollpunkt"}, value=row["role"], on_change=lambda event, name=str(row["name"]): save_role(name, event.value)).classes("w-48")

        @ui.refreshable
        def workspace() -> None:
            if (control_directory / "detections.json").is_file():
                render_review_workspace()
                return
            control_points = points(); image_root = project_directory / "products" / "dense" / "images"
            if control_points is None: return
            if not image_root.is_dir():
                ui.label("Entzerrte Bilder wurden zum Speicher sparen entfernt.").classes("sam-muted q-mt-md"); return
            images = sorted(path.relative_to(image_root).as_posix() for path in image_root.rglob("*") if path.is_file())
            if not images: ui.label("No undistorted images are available yet.").classes("sam-muted q-mt-md"); return
            selected["name"] = selected["name"] or str(control_points.iloc[0]["name"]); selected["image"] = selected["image"] or images[0]
            ui.label("Mark control points").classes("text-lg font-medium q-mt-lg")
            with ui.row().classes("w-full gap-4 items-start"):
                with ui.column().classes("w-72"):
                    ui.select(control_points["name"].tolist(), label="Active control point", value=selected["name"], on_change=lambda event: (selected.update(name=event.value), workspace.refresh()))
                    ui.select(images, label="Undistorted image", value=selected["image"], on_change=lambda event: (selected.update(image=event.value), workspace.refresh())).classes("w-full")
                    def remove() -> None:
                        current = clicks(); name, image = str(selected["name"]), str(selected["image"])
                        current[name] = [click for click in current.get(name, []) if click["image"] != image]
                        save_observations(observations_path, current); table.refresh(); workspace.refresh()
                    ui.button("Remove marker", on_click=remove).props("outline").classes("sam-button-secondary")
                with ui.column().classes("flex-1"):
                    ui.label("Click to set or move the active marker. Use browser image controls to zoom.").classes("sam-muted text-sm")
                    def mark(event) -> None:
                        u, v = getattr(event, "image_x", None), getattr(event, "image_y", None)
                        if u is None or v is None: ui.notify("Could not read the image click position.", type="negative"); return
                        current = clicks(); name, image = str(selected["name"]), str(selected["image"])
                        current[name] = [click for click in current.get(name, []) if click["image"] != image]
                        current[name].append({"image": image, "u": float(u), "v": float(v)})
                        save_observations(observations_path, current); table.refresh(); workspace.refresh()
                    ui.interactive_image("/api/control-point-image/" + quote(str(selected["image"]), safe="/"), on_mouse=mark).classes("w-full").style("max-height:70vh;object-fit:contain")
            usable = sum(len(clicks().get(str(name), [])) >= int(settings["min_observations_per_point"]) for name in control_points["name"])
            button = ui.button("Continue processing", icon="play_arrow", on_click=lambda: _continue(project_directory)).classes("sam-button-primary q-mt-md")
            button.set_enabled(usable >= int(settings["min_points"]))
            def check_control_points() -> None:
                try:
                    dense = project_directory / "products" / "dense"
                    _create_pristine_gcp_backups(dense / "sparse", dense / "sparse_txt", dense / "sparse_pre_gcp", dense / "sparse_txt_pre_gcp")
                    _, preview["report"], _ = georeference_from_control_points(
                        dense / "sparse_txt_pre_gcp", csv_path, observations_path, settings["crs"], settings,
                    )
                    preview["error"] = ""
                except (OSError, ValueError) as error:
                    preview["report"] = None; preview["error"] = str(error)
                preview_panel.refresh()
            ui.button("Check control points", icon="rule", on_click=check_control_points).props("outline").classes("sam-button-secondary q-mt-md q-ml-sm")

        @ui.refreshable
        def preview_panel() -> None:
            if preview["error"]:
                ui.label(preview["error"]).classes("text-red-4 text-sm q-mt-md")
                return
            report = preview["report"]
            if report is None:
                return
            threshold = float(settings.get("max_residual_warning_m", 0.10))
            ui.label("Control-point check").classes("text-lg font-medium q-mt-md")
            for point in report["points"]:
                warning = " warning" if point["residual_m"] > threshold else ""
                leave_one_out = point.get("leave_one_out_residual_m")
                suffix = f", leave-one-out {leave_one_out:.3f} m" if leave_one_out is not None else ""
                ui.label(f"{point['name']}: residual {point['residual_m']:.3f} m{suffix}{warning}").classes("text-red-4" if warning else "sam-muted")
        import_table(); table(); workspace(); preview_panel()
    render_page("Control points", content, current_path="/control-points", wide=True)


def _continue(project_directory: Path) -> None:
    try:
        state.processing.continue_after_control_points(project_directory)
        ui.notify("Control-point georeferencing and processing started."); ui.navigate.to("/")
    except (OSError, ValueError) as error:
        ui.notify(str(error), type="negative")
