# Spec: Punktwolke per QR-Code teilen (statischer Viewer)

Stand 2026-11-20 · Spezifikation des Auftraggebers · zur Umsetzung durch die Haupt-Session (Worker: Codex/Sonnet nach CLAUDE.md)

## Ziel
Auf der Punktwolken-Seite ("Dichte Punktwolke") steht direkt hinter dem Titel ein Knopf **„Teilen (QR)“**. Ein Klick:
1. exportiert die klassifizierte Punktwolke als **statischen Potree-Viewer** (HTML + JS + Octree),
2. lädt das Paket zu einem Hoster hoch,
3. zeigt währenddessen einen **Ladebalken** mit Phasen,
4. zeigt am Ende einen **QR-Code** + Link (kopierbar, PNG speicherbar).

Mit dem Handy scannen → die Punktwolke öffnet sich im Browser, **auch ausserhalb des eigenen Netzwerks**, ohne dass SAMcloud oder der PC laufen muss.

## Wann verfügbar
- Knopf nur aktiv, wenn **Schritt 6 (Verfeinerung) fertig** ist (`workflow.refinement == "completed"` bzw. finales `products/classified/classified.ply` vorhanden).
- Sonst deaktiviert mit Tooltip „Verfügbar nach Schritt 6“.
- Gilt für alle Aufnahmetypen (Drohne, 360°, Handy; innen/aussen).

## Funktionsumfang des geteilten Viewers (Handy-tauglich)
- RGB / Klassen umschalten, Klassen einzeln ein-/ausblenden (Legende mit deutschen Klassennamen + Farben aus der Projekt-`classes.json`).
- Passpunkte/AprilTags ein-/ausblenden (falls vorhanden; gleiche Daten wie der Knopf „Passpunkte“ im lokalen Viewer).
- Touch-Navigation (Potree Orbit/Earth-Controls), Punktgrösse-Regler, „Ansicht zurücksetzen“.
- Kopfzeile: Projektname, Datum, Punktzahl, Hinweis „SAMcloud“.
- Kein Bearbeiten, keine Messwerkzeuge nötig (optional später).
- Funktioniert offline-unabhängig: alle Bibliotheken im Paket (kein CDN).

## Ablauf (Backend)
Phasen mit Fortschritt (für den Ladebalken), jede mit Prozent + Text:
1. **Vorbereiten** – Punktwolke laden, optional ausdünnen (Ziel ≤ 20 Mio. Punkte für Mobile, konfigurierbar; Voxel-Ausdünnung, Klassen bleiben erhalten).
2. **Octree bauen** – Potree-Format erzeugen (bestehende Potree-Konvertierung im Projekt wiederverwenden; prüfen, wie `samcloud/app/potree_viewer.py` den lokalen Viewer erzeugt). Attribute: RGB, Klassifikation (LAS-Code), optional Intensität = Konfidenz.
3. **Viewer zusammenstellen** – statisches `index.html` aus einer Vorlage (Basis: lokaler Viewer, abgespeckt), `classes.json`-Legende, Passpunkt-Marker (JSON), Metadaten.
4. **Hochladen** – über einen austauschbaren Provider (siehe unten), mit Byte-Fortschritt.
5. **QR erzeugen** – Paket `segno` oder `qrcode` (falls nicht installiert: Rückfrage an den Auftraggeber, sonst reine Python-Implementierung vermeiden); PNG + SVG in `products/share/`.

- Läuft als Hintergrundjob (nicht im NiceGUI-Eventloop blockieren), Muster wie `ProcessingManager` in `samcloud/app/processing.py`.
- Fortschritt in `reports/share_progress.json`; UI pollt wie die bestehende Pipeline-Fortschrittsanzeige.
- Abbrechen-Knopf während des Jobs.
- Ergebnis gespeichert in `project.samcloud.yml` → `share: {url, created_at, provider, points, size_mb, expires_at}`; beim nächsten Öffnen wird der bestehende QR direkt angezeigt (mit „Neu erzeugen“ und „Link löschen“).

## Hosting (Provider-Schnittstelle)
`samcloud/share/providers/` mit gemeinsamer Schnittstelle `upload(local_dir, share_id, progress_cb) -> url` und `delete(share_id)`.
- **Erste Umsetzung: offen – Entscheidung des Auftraggebers** (siehe Offene Punkte). Vorschlag: **Cloudflare R2 + öffentliche Bucket-Domain** (statisch, günstig, kein Server) oder **Cloudflare Pages** per API.
- Zusätzlich ein **Provider „lokal“** für Tests/Entwicklung (kopiert ins Verzeichnis, URL `file://` bzw. lokaler HTTP-Server) – damit ist alles ohne Konto testbar.
- Zugangsdaten **nie im Repo**: aus Umgebungsvariablen bzw. Windows-Credential-Store; Konfiguration in `config/share.yaml` (ohne Secrets).

## Datenschutz / Sicherheit
- Link mit nicht erratbarer ID (≥ 128 Bit, z. B. `secrets.token_urlsafe(16)`).
- Vor dem ersten Hochladen pro Projekt ein Bestätigungsdialog: „Die Punktwolke wird öffentlich über diesen Link erreichbar. Fortfahren?“ (Hochladen = Veröffentlichen).
- Optional (Konfig): Ablaufdatum (Standard 30 Tage, Löschjob), Passwortschutz falls Provider es unterstützt (z. B. Cloudflare Access).
- Keine Bilder, keine GPS-Rohdaten, keine Projektpfade im Paket – nur Punktwolke, Klassen, Passpunkt-Namen/Positionen, Projektname.

## UI (Punktwolken-Seite)
- Knopf rechts neben dem Titel „Dichte Punktwolke“: Icon QR + „Teilen“.
- Klick → Dialog: Bestätigung (erstes Mal) → Ladebalken mit Phasennamen („Ausdünnen…“, „Octree bauen…“, „Hochladen 42 %…“) → QR-Code gross, darunter Link, Knöpfe „Link kopieren“, „QR speichern“, „Im Browser öffnen“, „Link löschen“.
- Texte über das bestehende i18n-System (Deutsch zuerst; andere vorhandene Sprachen mitpflegen).
- Design im Stil der bestehenden Toolbar (dunkles Theme, gleiche Button-Klasse).

## Dateien (Vorschlag, Überblick vorher mit Graphify bestätigen)
- neu: `samcloud/share/__init__.py`, `export.py` (Ausdünnen + Octree + Viewer-Paket), `viewer_template/` (index.html, JS, CSS), `providers/{base,local,cloudflare_r2}.py`, `qr.py`, `job.py` (Hintergrundjob + Fortschritt)
- geändert: Punktwolken-Seite (Knopf + Dialog; Datei der Seite ermitteln, vermutlich `samcloud/app/potree_viewer.py`), `samcloud/app/processing.py` oder eigener Manager für den Job, i18n-Dateien, `requirements.txt` (QR-Paket, ggf. `boto3` für R2 – nur nach Freigabe)
- Tests: `tests/test_share_export.py`, `test_share_job.py`, `test_share_providers.py`, `test_share_ui.py`

## Akzeptanzkriterien
1. Knopf nur nach Schritt 6 aktiv; sonst deaktiviert mit Tooltip.
2. Lokaler Provider: Export eines kleinen Testprojekts erzeugt ein Paket, das in einem Browser ohne SAMcloud öffnet (per Preview prüfen), RGB/Klassen-Umschaltung und Klassen-Legende funktionieren.
3. Ladebalken zeigt alle Phasen, Abbrechen funktioniert, Fehler werden als Meldung angezeigt (kein hängender Dialog).
4. QR-PNG decodiert (Test mit einer QR-Lese-Bibliothek oder OpenCV `QRCodeDetector`) zum gespeicherten Link.
5. Mobile-Ansicht (Viewport 375×812) bedienbar; Punktwolke mit ≤ 20 Mio. Punkten lädt.
6. Keine Secrets im Repo/Paket; Paket enthält keine Bilder/Pfade.
7. Bestehende Viewer-/Potree-Tests grün, volle Suite grün.
8. Echter Upload erst nach Freigabe von Provider + Konto durch den Auftraggeber; Test mit dem Drohnenprojekt `drohnenrekonstruktion_3d_test`, Scan mit dem Handy durch den Auftraggeber.

## Offene Punkte (der Auftraggeber entscheidet)
1. **Hoster/Konto:** Cloudflare (R2 oder Pages), GitHub Pages, HSLU-Server oder anderes?
2. **Zugriff:** frei mit geheimem Link, oder Passwort/Ablaufdatum?
3. **Max. Punktzahl** für die Teilen-Version (Vorschlag 20 Mio.).
