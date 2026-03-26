#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
JSON → PDF Conversion with Corporate Styling
============================================
Compatible with tickets_to_json_claude.py (fixed schema).

Changes vs previous version:
  - facts.societe              : new company name field
  - chronologie[].acteur       : colour-coded actor label (client/support/système)
  - probleme.symptomes         : now rendered
  - cause_racine.composants_affectes : now rendered
  - patches[]                  : new section (grid layout, all patches)
  - prerequis[]                : new section
  - erreurs_documentation[]    : new section ({document, erreur, correction} dicts)
  - resolution.actions_realisees : handles both list (new) and string (old)
  - HRFlowable separators in chronology
  - CellMono style for patch numbers
  - --skip-existing CLI flag
  - All prior bug fixes retained (fermeture, durée, responsabilité, statut)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from xml.sax.saxutils import escape as xml_escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


# ============================================================================
# HELPERS
# ============================================================================

def esc(x: Any) -> str:
    """Escape text for ReportLab paragraphs."""
    if x is None:
        return ""
    if isinstance(x, list):
        return ", ".join(esc(i) for i in x)
    return xml_escape(str(x))


def clean_keywords(words: List[str]) -> List[str]:
    clean = []
    for w in words:
        if not w:
            continue
        w = w.replace("\x7f", "").strip()
        if w and w != "non_specifie":
            clean.append(w)
    return clean


def get_severity_color(gravite: str):
    g = str(gravite or "").lower()
    if "block" in g:
        return colors.HexColor("#dc3545")
    if "important" in g or "très" in g:
        return colors.HexColor("#fd7e14")
    if "serious" in g or "sérieux" in g:
        return colors.HexColor("#ffc107")
    return colors.HexColor("#198754")


def acteur_color(acteur: str) -> str:
    a = str(acteur or "").lower()
    if "support" in a:
        return "#0b5ed7"
    if "client" in a:
        return "#198754"
    return "#6b7280"


def format_actions(actions_realisees: Any) -> str:
    """Handle both list (new schema) and string (old schema)."""
    if isinstance(actions_realisees, list):
        items = [str(a) for a in actions_realisees if a and str(a) != "non_specifie"]
        return "\n".join(f"• {a}" for a in items) if items else "N/A"
    if isinstance(actions_realisees, str) and actions_realisees:
        return actions_realisees
    return "N/A"


def format_erreurs_documentation(erreurs: Any) -> List[Dict]:
    """Handle both list-of-dicts (new) and list-of-strings (old)."""
    if not erreurs:
        return []
    result = []
    for e in erreurs:
        if isinstance(e, dict):
            result.append(e)
        elif isinstance(e, str) and e and e != "non_specifie":
            result.append({
                "document":   "non_specifie",
                "erreur":     e,
                "correction": "non_specifie",
            })
    return result


# ============================================================================
# STYLES
# ============================================================================

def create_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="CustomTitle",
        parent=styles["Heading1"],
        fontSize=22,
        alignment=TA_CENTER,
        spaceAfter=18,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0b5ed7"),
    ))

    styles.add(ParagraphStyle(
        name="Subtitle",
        fontSize=12,
        alignment=TA_CENTER,
        spaceAfter=10,
        textColor=colors.HexColor("#374151"),
    ))

    styles.add(ParagraphStyle(
        name="Heading2Custom",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=14,
        spaceAfter=8,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#0b5ed7"),
    ))

    styles.add(ParagraphStyle(
        name="BodyCustom",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=14.5,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        textColor=colors.HexColor("#1f2933"),
    ))

    styles.add(ParagraphStyle(
        name="Small",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#4b5563"),
    ))

    styles.add(ParagraphStyle(
        name="Cell",
        fontSize=9.5,
        leading=13,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#1f2933"),
    ))

    styles.add(ParagraphStyle(
        name="CellMono",
        fontSize=8.5,
        leading=12,
        alignment=TA_LEFT,
        fontName="Courier",
        textColor=colors.HexColor("#1f2933"),
    ))

    styles.add(ParagraphStyle(
        name="Label",
        fontSize=9.5,
        leading=13,
        alignment=TA_LEFT,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#111827"),
    ))

    styles.add(ParagraphStyle(
        name="Footer",
        fontSize=8,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#6b7280"),
    ))

    return styles


# ============================================================================
# CORE: JSON → PDF
# ============================================================================

def json_to_pdf(json_path: Path, pdf_path: Path) -> bool:

    # ── Safe JSON loading ──────────────────────────────────────────────────
    try:
        raw = json_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = json_path.read_text(encoding="latin-1")
        raw = raw.encode("utf-8", errors="replace").decode("utf-8")

    try:
        data = json.loads(raw)
    except Exception as e:
        logging.error(f"[JSON ERROR] {json_path.name}: {e}")
        return False

    try:
        # ── Extract fields ─────────────────────────────────────────────────
        ref   = data.get("reference", "UNKNOWN")
        facts = data.get("facts", {}) or {}

        societe   = facts.get("societe", "")           # new — absent in old JSONs
        client    = facts.get("client", "")
        version   = facts.get("version", "")
        systeme   = facts.get("systeme", facts.get("system", ""))
        ouverture = facts.get("ouverture", "")
        fermeture = facts.get("fermeture", "")
        duree     = facts.get("duree", "")

        equipe         = data.get("equipe_responsable", "non_specifie")
        probleme       = data.get("probleme", {}) or {}
        cause          = data.get("cause_racine", {}) or {}
        chron          = data.get("chronologie", []) or []
        reso           = data.get("resolution", {}) or {}

        statut_reso    = reso.get("statut", "N/A")
        code_reso      = reso.get("code", "N/A")
        verdict        = reso.get("verdict", "N/A")
        responsabilite = reso.get("responsabilite", "N/A")
        actions_str    = format_actions(reso.get("actions_realisees", ""))

        patches   = [str(p) for p in (data.get("patches", []) or [])
                     if str(p) != "non_specifie"]
        prerequis = [str(p) for p in (data.get("prerequis", []) or [])
                     if str(p) != "non_specifie"]
        erreurs   = format_erreurs_documentation(data.get("erreurs_documentation", []))
        impact    = data.get("impact_metier", "Impact non précisé") or ""
        mots_cles = clean_keywords(data.get("mots_cles", []) or [])
        meta      = data.get("_metadata", {}) or {}

        # ── PDF setup ──────────────────────────────────────────────────────
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=16 * mm,
            leftMargin=16 * mm,
            topMargin=18 * mm,
            bottomMargin=22 * mm,
        )

        styles = create_styles()
        story: List[Any] = []

        # ── Header ─────────────────────────────────────────────────────────
        story.append(Paragraph(f"🎫 TICKET {esc(ref)}", styles["CustomTitle"]))
        story.append(Paragraph("Enterprise Incident Analysis &amp; Resolution", styles["Subtitle"]))
        story.append(Paragraph(
            f"<font size='9'>"
            f"Generated: {esc(meta.get('generated_at', 'N/A'))}<br/>"
            f"Model: {esc(meta.get('model', 'N/A'))}"
            f"</font>",
            styles["Small"],
        ))
        story.append(Spacer(1, 10))

        # ── General info ───────────────────────────────────────────────────
        story.append(Paragraph("📋 Informations générales", styles["Heading2Custom"]))

        facts_rows = [
            ("Société",                 societe),
            ("Client",                  client),
            ("Version applicative",     version),
            ("Système / Environnement", systeme),
            ("Ouverture",               ouverture),
            ("Fermeture",               fermeture),
            ("Durée",                   duree),
            ("Équipe responsable",      equipe),
        ]
        # Skip rows with no value (e.g. societe absent in old JSONs)
        facts_rows = [(k, v) for k, v in facts_rows if v and v != "non_specifie"]

        table_data = [["Information", "Détail"]]
        for label, val in facts_rows:
            table_data.append([
                Paragraph(esc(label), styles["Label"]),
                Paragraph(esc(val),   styles["Cell"]),
            ])

        t = Table(table_data, colWidths=[65 * mm, 100 * mm])
        t.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0),  colors.HexColor("#f3f6fa")),
            ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#cfd8e3")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f9fafb")]),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

        # ── Problem ────────────────────────────────────────────────────────
        story.append(Paragraph("❗ Problème", styles["Heading2Custom"]))

        story.append(Paragraph("<b>📌 Titre</b>", styles["Small"]))
        story.append(Paragraph(esc(probleme.get("titre", "N/A")), styles["BodyCustom"]))

        story.append(Paragraph("<b>📝 Description</b>", styles["Small"]))
        story.append(Paragraph(esc(probleme.get("description", "N/A")), styles["BodyCustom"]))

        gravite = probleme.get("gravite", "")
        gcol    = get_severity_color(gravite)
        story.append(Paragraph("<b>🔴 Gravité</b>", styles["Small"]))
        story.append(Paragraph(
            f"<font color='{gcol.hexval()}'><b>{esc(gravite)}</b></font>",
            styles["BodyCustom"],
        ))

        # Symptomes — new field, gracefully absent in old JSONs
        symptomes = [s for s in (probleme.get("symptomes") or [])
                     if s and s != "non_specifie"]
        if symptomes:
            story.append(Paragraph("<b>🩺 Symptômes</b>", styles["Small"]))
            for s in symptomes:
                story.append(Paragraph(f"• {esc(s)}", styles["BodyCustom"]))

        story.append(Spacer(1, 10))

        # ── Root cause ─────────────────────────────────────────────────────
        story.append(Paragraph("🧠 Cause racine", styles["Heading2Custom"]))

        story.append(Paragraph("<b>💡 Explication synthétique</b>", styles["Small"]))
        story.append(Paragraph(esc(cause.get("explication", "N/A")), styles["BodyCustom"]))

        story.append(Paragraph("<b>🏷️ Type d'incident</b>", styles["Small"]))
        story.append(Paragraph(esc(cause.get("type_incident", "N/A")), styles["BodyCustom"]))

        # Composants affectés — new field, gracefully absent in old JSONs
        composants = [c for c in (cause.get("composants_affectes") or [])
                      if c and c != "non_specifie"]
        if composants:
            story.append(Paragraph("<b>⚙️ Composants affectés</b>", styles["Small"]))
            story.append(Paragraph(
                ", ".join(esc(c) for c in composants), styles["BodyCustom"]
            ))

        story.append(Spacer(1, 10))

        # ── Chronology ─────────────────────────────────────────────────────
        story.append(Paragraph("🕒 Chronologie du diagnostic", styles["Heading2Custom"]))

        for step in chron:
            date   = step.get("date", "")
            acteur = step.get("acteur", "")      # new — absent in old JSONs
            action = step.get("action", "")
            desc   = step.get("description", "")
            res    = step.get("resultat", "")
            hyp    = step.get("hypothese", "")   # old field — tolerated if present

            # Date header with optional colour-coded actor label
            if acteur:
                acol   = acteur_color(acteur)
                header = (
                    f"<b>{esc(date)}</b>"
                    f" — <font color='{acol}'><b>{esc(acteur.upper())}</b></font>"
                )
            else:
                header = f"<b>{esc(date)}</b>"

            story.append(Paragraph(header, styles["BodyCustom"]))

            if action:
                story.append(Paragraph(f"<b>Action :</b> {esc(action)}", styles["BodyCustom"]))
            if desc:
                story.append(Paragraph(f"<b>Description :</b> {esc(desc)}", styles["BodyCustom"]))
            if hyp:
                story.append(Paragraph(f"<b>Hypothèse :</b> {esc(hyp)}", styles["BodyCustom"]))
            if res:
                story.append(Paragraph(f"<b>Résultat :</b> {esc(res)}", styles["BodyCustom"]))

            story.append(HRFlowable(
                width="100%", thickness=0.3,
                color=colors.HexColor("#e5e7eb"), spaceAfter=6,
            ))

        # ── Resolution ─────────────────────────────────────────────────────
        story.append(Paragraph("✅ Résolution", styles["Heading2Custom"]))

        res_rows = [
            ("Statut",         statut_reso),
            ("Code",           code_reso),
            ("Verdict",        verdict),
            ("Responsabilité", responsabilite),
        ]
        res_rows = [(k, v) for k, v in res_rows
                    if v and v not in ("N/A", "non_specifie")]

        if res_rows:
            res_data = [["Métrique", "Détail"]]
            for k, v in res_rows:
                res_data.append([
                    Paragraph(esc(k), styles["Label"]),
                    Paragraph(esc(v), styles["Cell"]),
                ])
            t_res = Table(res_data, colWidths=[60 * mm, 105 * mm])
            t_res.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f6fa")),
                ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#cfd8e3")),
            ]))
            story.append(t_res)

        if actions_str and actions_str != "N/A":
            story.append(Spacer(1, 6))
            story.append(Paragraph("<b>Actions réalisées</b>", styles["Small"]))
            story.append(Paragraph(esc(actions_str), styles["BodyCustom"]))

        story.append(Spacer(1, 10))

        # ── Patches ────────────────────────────────────────────────────────
        if patches:
            story.append(Paragraph("🔧 Patches", styles["Heading2Custom"]))

            row_size   = 8
            patch_rows = [patches[i:i + row_size]
                          for i in range(0, len(patches), row_size)]
            patch_data = [
                [Paragraph(esc(p), styles["CellMono"]) for p in row]
                for row in patch_rows
            ]

            # Pad the last row so column widths stay consistent
            if patch_data and len(patch_data[-1]) < row_size:
                pad = row_size - len(patch_data[-1])
                patch_data[-1] += [Paragraph("", styles["CellMono"])] * pad

            t_patches = Table(patch_data, colWidths=[20 * mm] * row_size)
            t_patches.setStyle(TableStyle([
                ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
                ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("TOPPADDING",    (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(t_patches)
            story.append(Paragraph(
                f"<font size='8' color='#6b7280'>{len(patches)} patch(es) au total</font>",
                styles["Small"],
            ))
            story.append(Spacer(1, 10))

        # ── Prerequisites ──────────────────────────────────────────────────
        if prerequis:
            story.append(Paragraph("📦 Prérequis", styles["Heading2Custom"]))
            for p in prerequis:
                story.append(Paragraph(f"• {esc(p)}", styles["BodyCustom"]))
            story.append(Spacer(1, 10))

        # ── Documentation errors ───────────────────────────────────────────
        if erreurs:
            story.append(Paragraph("📄 Erreurs de documentation", styles["Heading2Custom"]))

            err_data = [["Document", "Erreur", "Correction"]]
            for e in erreurs:
                doc_name = e.get("document", "") or ""
                erreur   = e.get("erreur",    "") or ""
                correct  = e.get("correction","") or ""
                if not erreur or erreur == "non_specifie":
                    continue
                err_data.append([
                    Paragraph(esc(doc_name), styles["Cell"]),
                    Paragraph(esc(erreur),   styles["Cell"]),
                    Paragraph(esc(correct),  styles["Cell"]),
                ])

            if len(err_data) > 1:
                t_err = Table(err_data, colWidths=[40 * mm, 75 * mm, 50 * mm])
                t_err.setStyle(TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#fff3cd")),
                    ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cfd8e3")),
                    ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
                    ("ROWBACKGROUNDS",(0, 1), (-1, -1),
                     [colors.white, colors.HexColor("#fffdf0")]),
                ]))
                story.append(t_err)
            story.append(Spacer(1, 10))

        # ── Business impact ────────────────────────────────────────────────
        story.append(Paragraph("💼 Impact métier", styles["Heading2Custom"]))
        story.append(Paragraph(esc(impact), styles["BodyCustom"]))

        # ── Keywords ───────────────────────────────────────────────────────
        if mots_cles:
            story.append(Paragraph("🏷️ Mots-clés", styles["Heading2Custom"]))
            story.append(Paragraph(
                " • ".join(esc(k) for k in mots_cles), styles["BodyCustom"]
            ))

        story.append(Spacer(1, 20))
        story.append(Paragraph(
            "Document généré automatiquement — Confidentiel",
            styles["Footer"],
        ))

        doc.build(story)
        return True

    except Exception as e:
        logging.error(f"[PDF ERROR] {json_path.name}: {e}", exc_info=True)
        return False


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Convert Sopra HR analysis JSON to PDF.")
    ap.add_argument("-i", "--input",       required=True, type=Path,
                    help="Folder containing *_analysis.json files")
    ap.add_argument("-o", "--output",      required=True, type=Path,
                    help="Output folder for PDF files")
    ap.add_argument("-v", "--verbose",     action="store_true")
    ap.add_argument("--skip-existing",     action="store_true",
                    help="Skip tickets that already have a PDF")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    args.output.mkdir(parents=True, exist_ok=True)

    sources = sorted(args.input.glob("*_analysis.json"))
    logging.info(f"Found {len(sources)} analysis files")

    success = 0
    for jf in sources:
        pdf_path = args.output / f"{jf.stem.replace('_analysis', '')}_analysis.pdf"

        if args.skip_existing and pdf_path.exists():
            logging.info(f"[SKIP] {pdf_path.name}")
            continue

        if json_to_pdf(jf, pdf_path):
            logging.info(f"[PDF]  {pdf_path.name}")
            success += 1

    logging.info(f"✔ Completed {success} / {len(sources)} PDF files.")


if __name__ == "__main__":
    main()