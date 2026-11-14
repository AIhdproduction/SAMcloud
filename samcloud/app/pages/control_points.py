"""Ground-control-point import and image-marking workspace."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from nicegui import events, ui

from samcloud.app.layout import render_page
from samcloud.app.pages.gcp_review import render_review_workspace
from samcloud.app.state import state
from samcloud.control_points.repository import APRILTAG_CSV_NAME, apply_roles, read_apriltag_coordinates, load_observations, read_control_points, save_observations, write_control_points
from samcloud.control_points.apriltag_print import build_apriltag_pdf
from samcloud.control_points.apriltags import APRILTAG_ID_MAX
from samcloud.control_points.georeference import georeference_from_control_points
from samcloud.core.project_manager import LOCAL_CRS_GPS_MESSAGE, set_apriltag_coordinates
from samcloud.engine.pipeline import _create_pristine_gcp_backups


_CRS_OPTIONS = {
    "EPSG:2056+5728": "LV95 + LN02", "EPSG:2056": "LV95 ellipsoidal", "EPSG:4979": "WGS84 ellipsoidal",
    "LOCAL": "Lokal (eigenes System, Meter)",
}
LOCAL_CRS_HINT = "Lokal: Die Punktwolke wird direkt in diese Koordinaten gelegt (Meter). Kein Export in LV95/WGS84 möglich."


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
            local_hint.set_visibility(event.value == "LOCAL")

        def save_role(name: str, role: str) -> None:
            control_points = points()
            if control_points is not None:
                write_control_points(apply_roles(control_points, {name: role}), csv_path)
                table.refresh()

        with ui.card().classes("w-full"):
            ui.label("Control point coordinates").classes("text-lg font-medium")
            ui.label("Import name, x, y, z coordinates. The default is Swiss LV95 + LN02.").classes("sam-muted text-sm")
            ui.select(_CRS_OPTIONS, label="Control-point CRS", value=settings["crs"], on_change=save_crs).classes("w-full max-w-md").mark("control-points-crs")
            local_hint = ui.label(LOCAL_CRS_HINT).classes("sam-muted text-sm").mark("control-points-crs-local-hint")
            local_hint.set_visibility(settings["crs"] == "LOCAL")
            async def import_csv(event: events.UploadEventArguments) -> None:
                upload = control_directory / ".uploaded_control_points"
                await event.file.save(upload)
                try:
                    pending_import["points"] = read_control_points(upload)
                except ValueError as error: ui.notify(str(error), type="negative")
                import_table.refresh()
            ui.upload(on_upload=import_csv, auto_upload=True, label="Punktdatei importieren").props("accept=.csv,.txt,.dat").classes("q-mt-md")
        with ui.card().classes("w-full"):
            ui.label("AprilTags").classes("text-lg font-medium")
            ui.label("150 mm Tags (tag36h11). Mit 100 % drucken, Kontrollstrecke nachmessen. Mindestens 3 Tags pro Projekt empfohlen.").classes("sam-muted text-sm")
            tag_settings = project.setdefault("settings", {}).setdefault("apriltags", {})
            tag_settings.setdefault("last_printed_id", -1)
            with ui.dialog() as tag_dialog, ui.card():
                ui.label("AprilTags drucken").classes("text-lg font-medium")
                tag_count = ui.number("Anzahl", value=10, min=1, max=100, step=1, precision=0).mark("apriltag-count")
                tag_start = ui.number("Start-ID", value=int(tag_settings["last_printed_id"]) + 1, min=0, max=APRILTAG_ID_MAX, step=1, precision=0).mark("apriltag-start")

                def create_tag_pdf() -> None:
                    try:
                        start, count = int(tag_start.value), int(tag_count.value)
                        pdf = build_apriltag_pdf(start, count)
                    except (TypeError, ValueError) as error:
                        ui.notify(str(error), type="negative")
                        return
                    end = start + count - 1
                    ui.download(pdf, filename=f"apriltags_{start}-{end}.pdf", media_type="application/pdf")
                    tag_settings["last_printed_id"] = end
                    state.projects.save(project_directory, project)
                    tag_dialog.close()

                with ui.row().classes("justify-end w-full gap-2"):
                    ui.button("Abbrechen", on_click=tag_dialog.close).props("flat")
                    ui.button("PDF erstellen", on_click=create_tag_pdf).classes("sam-button-primary").mark("apriltag-pdf")

            def open_tag_dialog() -> None:
                tag_start.value = int(tag_settings.get("last_printed_id", -1)) + 1
                tag_dialog.open()

            ui.button("AprilTag drucken", icon="print", on_click=open_tag_dialog).classes("sam-button-primary q-mt-sm").mark("apriltag-print")
            ui.separator().classes("q-my-md")
            ui.label("Tag-Koordinaten").classes("text-base font-medium")
            ui.label("Optional: Koordinaten der Tag-Mitte je ID. Die ID auf dem Ausdruck muss mit der ID in der Datei übereinstimmen.").classes("sam-muted text-sm")
            tag_csv_path = control_directory / APRILTAG_CSV_NAME
            tag_state: dict[str, list[dict]] = {"rows": []}
            if tag_csv_path.is_file():
                try:
                    tag_state["rows"] = [{"id": int(row["id"]), "x": float(row["x"]), "y": float(row["y"]), "z": float(row["z"]), "role": str(row["role"]) or "pass"} for row in read_apriltag_coordinates(tag_csv_path).to_dict("records")]
                except (OSError, ValueError) as error:
                    ui.notify(str(error), type="negative")

            @ui.refreshable
            def tag_coords() -> None:
                rows = sorted(tag_state["rows"], key=lambda row: row["id"])
                columns = [{"name": key, "label": label, "field": key, "align": "left"} for key, label in (("id", "ID"), ("x", "X"), ("y", "Y"), ("z", "Z"), ("role", "Rolle"))]
                coords_table = ui.table(columns=columns, rows=rows, row_key="id", selection="single").classes("w-full q-mt-sm").mark("apriltag-coords-table")

                def load_selected() -> None:
                    chosen = coords_table.selected
                    if chosen:
                        edit_id.value, edit_x.value, edit_y.value, edit_z.value, edit_role.value = chosen[0]["id"], chosen[0]["x"], chosen[0]["y"], chosen[0]["z"], chosen[0]["role"]
                coords_table.on("selection", lambda _: load_selected())
                with ui.row().classes("items-center gap-2 q-mt-sm"):
                    edit_id = ui.number("ID", min=0, max=APRILTAG_ID_MAX, step=1, precision=0).classes("w-24").mark("apriltag-coords-id")
                    edit_x = ui.number("X", format="%.4f").classes("w-36").mark("apriltag-coords-x")
                    edit_y = ui.number("Y", format="%.4f").classes("w-36").mark("apriltag-coords-y")
                    edit_z = ui.number("Z", format="%.4f").classes("w-36").mark("apriltag-coords-z")
                    edit_role = ui.select({"pass": "pass", "control": "control"}, value="pass", label="Rolle").classes("w-32").mark("apriltag-coords-role")

                    def upsert_row() -> None:
                        if None in (edit_id.value, edit_x.value, edit_y.value, edit_z.value):
                            ui.notify("ID, X, Y und Z angeben", type="negative")
                            return
                        entry = {"id": int(edit_id.value), "x": float(edit_x.value), "y": float(edit_y.value), "z": float(edit_z.value), "role": edit_role.value or "pass"}
                        tag_state["rows"] = [row for row in tag_state["rows"] if row["id"] != entry["id"]] + [entry]
                        tag_coords.refresh()

                    def remove_row() -> None:
                        if edit_id.value is not None:
                            tag_state["rows"] = [row for row in tag_state["rows"] if row["id"] != int(edit_id.value)]
                            tag_coords.refresh()

                    ui.button("Zeile übernehmen", on_click=upsert_row).props("outline").mark("apriltag-coords-apply")
                    ui.button("Zeile entfernen", on_click=remove_row).props("outline").mark("apriltag-coords-remove")
                if tag_state["rows"] and sum(1 for row in tag_state["rows"] if row["role"] == "pass") < 3:
                    ui.label("Für die Georeferenzierung mindestens 3 eingemessene Tags (nicht auf einer Linie) – sonst nur Maßstab").classes("text-red-4 text-sm q-mt-sm").mark("apriltag-coords-warning")

            tag_coords()

            async def import_tag_coords(event: events.UploadEventArguments) -> None:
                control_directory.mkdir(parents=True, exist_ok=True)
                upload = control_directory / ".uploaded_apriltags"
                await event.file.save(upload)
                try:
                    imported = read_apriltag_coordinates(upload)
                except ValueError as error:
                    ui.notify(str(error), type="negative")
                    return
                tag_state["rows"] = [{"id": int(row["id"]), "x": float(row["x"]), "y": float(row["y"]), "z": float(row["z"]), "role": str(row["role"]) or "pass"} for row in imported.to_dict("records")]
                tag_coords.refresh()

            def save_tag_coords() -> None:
                if not tag_state["rows"]:
                    ui.notify("Keine Tag-Koordinaten zum Speichern", type="negative")
                    return
                try:
                    set_apriltag_coordinates(project_directory, project, pd.DataFrame(tag_state["rows"]), settings["crs"])
                except ValueError as error:
                    ui.notify(str(error), type="negative")
                    return
                ui.notify("Tag-Koordinaten gespeichert")

            ui.upload(on_upload=import_tag_coords, auto_upload=True, label="Tag-Koordinaten importieren").props("accept=.csv,.txt").classes("q-mt-sm").mark("apriltag-coords-import")
            ui.button("Tag-Koordinaten speichern", icon="save", on_click=save_tag_coords).classes("sam-button-primary q-mt-sm").mark("apriltag-coords-save")
            tag_report_path = project_directory / "reports" / "apriltag_report.json"
            if tag_report_path.is_file():
                try:
                    tag_report = json.loads(tag_report_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    tag_report = None
                if isinstance(tag_report, dict):
                    scale_applied = tag_report.get("scale_applied")
                    ui.label("Maßstab aus AprilTags: " + (f"Faktor {float(scale_applied):.4f}" if scale_applied is not None else "nicht angewendet")).classes("q-mt-md")
                    for warning in tag_report.get("warnings", []):
                        ui.label(str(warning)).classes("text-red-4 text-sm")
                    tag_rows = [{"id": tag["id"], "images": tag["images"], "before": f"{float(tag['size_mm_before']):.1f}", "after": f"{float(tag['size_mm_after']):.1f}", "deviation": "" if tag.get("deviation_pct") is None else f"{float(tag['deviation_pct']):+.2f}", "status": tag.get("status", "")} for tag in tag_report.get("tags", [])]
                    tag_columns = [{"name": key, "label": label, "field": key} for key, label in (("id", "ID"), ("images", "Bilder"), ("before", "Größe vorher mm"), ("after", "Größe nachher mm"), ("deviation", "Abweichung %"), ("status", "Status"))]
                    ui.table(columns=tag_columns, rows=tag_rows, row_key="id").classes("w-full q-mt-sm").mark("apriltag-report")


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
                    if settings["crs"] == "LOCAL":
                        raise ValueError(LOCAL_CRS_GPS_MESSAGE)
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
