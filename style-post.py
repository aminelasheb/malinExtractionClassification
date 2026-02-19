import json
import csv
import os
import re
import difflib
import glob
from pathlib import Path

# --- FONCTIONS UTILITAIRES DE STYLE ---

def is_special_color(c):
    """Vérifie si la couleur est pertinente (ignore noir/blanc pour le filtrage final)."""
    if not c: return False
    c = c.strip().lower()
    return c.startswith('#') and c not in ['#181715', '#231f20', '#000000', '#ffffff', '#fff']


def is_black_color(c):
    """Détermine si une couleur est considérée comme 'standard' (Noir/Gris foncé)."""
    if not c: return True  # Pas de couleur = standard
    c = c.strip().lower()
    return c in ['#181715', '#231f20', '#000000']


def is_bold_style(s):
    if not s: return False
    s = s.strip().lower()
    return any(sub in s for sub in ['bold', 'medium', 'black', 'heavy'])


def is_italic_style(s):
    if not s: return False
    s = s.strip().lower()
    return any(sub in s for sub in ['italic', 'oblique', 'italique'])


# --- MOTEUR PRINCIPAL ---

def process_page(json_path, csv_path, output_path):
    if not os.path.exists(json_path) or not os.path.exists(csv_path):
        return

    # 1. CHARGEMENT CSV
    global_csv_chars = []
    global_csv_styles = []

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            phrase = row.get('phrase', '')
            overrides = row.get('overrides', '').strip()
            base_color = row.get('color_hex', '').strip()
            base_style = row.get('style_tag', '').strip()

            if not phrase: continue
            N = len(phrase)

            row_bold = is_bold_style(base_style)
            row_italic = is_italic_style(base_style)

            char_bold = [row_bold] * N
            char_italic = [row_italic] * N
            char_color = [base_color] * N
            is_marker = [False] * N

            if overrides:
                items = overrides.split('||')
                for item in items:
                    parts = item.split('|')
                    if len(parts) >= 5:
                        target = parts[0].strip()
                        if not target: continue
                        o_color = parts[3].strip()
                        o_style = parts[4].strip()
                        t_bold = is_bold_style(o_style)
                        t_italic = is_italic_style(o_style)

                        target_esc = re.escape(target)
                        if re.match(r'^\w', target, flags=re.UNICODE):
                            target_esc = r'(?<!\w)' + target_esc
                        if re.search(r'\w$', target, flags=re.UNICODE):
                            target_esc = target_esc + r'(?!\w)'

                        for m in re.finditer(target_esc, phrase, flags=re.UNICODE):
                            for i in range(m.start(), m.end()):
                                char_bold[i] = t_bold
                                char_italic[i] = t_italic
                                char_color[i] = o_color

            m_list = re.match(r'^\s*([a-zA-Z0-9]{1,3}[\.\)]|[●•\-–➝])\s*', phrase)
            if m_list:
                start, end = m_list.span(1)
                for i in range(start, end):
                    is_marker[i] = True

            for i in range(N):
                c = phrase[i]
                if not re.match(r'\s|\x07', c):
                    global_csv_chars.append(c)
                    global_csv_styles.append({
                        'bold': False if is_marker[i] else char_bold[i],
                        'italic': False if is_marker[i] else char_italic[i],
                        'color': None if is_marker[i] else char_color[i]
                    })

    global_csv_string = "".join(global_csv_chars)

    # 2. CHARGEMENT JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nodes = []

    def gather_strings(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ['instruction', 'statement', 'hint', 'example'] and isinstance(value, str):
                    nodes.append({'parent': node, 'key': key, 'text': value})
                elif key == 'labels' and isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, str):
                            nodes.append({'parent': value, 'key': i, 'text': item})
                else:
                    gather_strings(value)
        elif isinstance(node, list):
            for item in node: gather_strings(item)

    gather_strings(data)

    # 3. ALIGNEMENT
    cursor_hint = 0
    for node in nodes:
        text = node['text']
        json_chars = []
        json_indices = []
        for i, c in enumerate(text):
            if not re.match(r'\s|\x07', c):
                json_chars.append(c)
                json_indices.append(i)

        node_nw_string = "".join(json_chars)
        if not node_nw_string: continue

        matcher = difflib.SequenceMatcher(None, global_csv_string, node_nw_string, autojunk=False)
        match = matcher.find_longest_match(0, len(global_csv_string), 0, len(node_nw_string))

        N_json = len(text)
        final_bold = [False] * N_json
        final_italic = [False] * N_json
        final_color = [None] * N_json

        if match.size > 0:
            csv_start = match.a
            for i in range(match.size):
                if i < len(json_indices):
                    real_idx = json_indices[i]
                    style = global_csv_styles[csv_start + i]
                    final_bold[real_idx] = style['bold']
                    final_italic[real_idx] = style['italic']
                    final_color[real_idx] = style['color']
            cursor_hint = csv_start + match.size

        # 4. LISSAGE
        for i in range(1, N_json - 1):
            if re.match(r'\s|\x07', text[i]):
                l = i - 1
                while l >= 0 and re.match(r'\s|\x07', text[l]): l -= 1
                r = i + 1
                while r < N_json and re.match(r'\s|\x07', text[r]): r += 1
                if l >= 0 and r < N_json:
                    if final_bold[l] == final_bold[r] and final_italic[l] == final_italic[r] and final_color[l] == \
                            final_color[r]:
                        final_bold[i] = final_bold[l]
                        final_italic[i] = final_italic[l]
                        final_color[i] = final_color[l]

        # 5. LOGIQUE DELTA (GRAS / ITALIQUE / COULEUR)
        marker_len = 0
        m_marker = re.match(r'^\s*([a-zA-Z0-9]{1,3}[\.\)]|[●•\-–➝])\s*', text)
        if m_marker: marker_len = m_marker.end()

        valid_indices = []
        for i in range(marker_len, N_json):
            if not re.match(r'\s|\x07', text[i]):
                valid_indices.append(i)

        if valid_indices:
            # A. Delta Gras
            all_bold = all(final_bold[i] for i in valid_indices)
            if all_bold:
                for i in range(N_json): final_bold[i] = False

            # B. Delta Italique
            all_italic = all(final_italic[i] for i in valid_indices)
            if all_italic:
                for i in range(N_json): final_italic[i] = False

            # C. Delta COULEUR (INTELLIGENT)
            visible_colors = []
            for i in valid_indices:
                if final_color[i]: visible_colors.append(final_color[i].lower())

            if visible_colors:
                unique_colors = set(visible_colors)

                # 1. Si tout est uniforme (ex: Tout Mauve, ou Tout Noir) -> On nettoie tout
                if len(unique_colors) == 1:
                    for i in range(N_json): final_color[i] = None
                else:
                    # 2. Si c'est mélangé (ex: Mauve + Noir)
                    # On cherche si le NOIR est présent. Si oui, le Noir est la BASE.
                    has_black = any(is_black_color(c) for c in unique_colors)

                    if has_black:
                        # La base est le Noir. On nettoie le noir, on garde le reste.
                        for i in range(N_json):
                            if is_black_color(final_color[i]):
                                final_color[i] = None
                    else:
                        # Pas de noir (ex: Rouge + Bleu). La base est la couleur majoritaire.
                        from collections import Counter
                        most_common = Counter(visible_colors).most_common(1)[0][0]
                        for i in range(N_json):
                            if final_color[i] and final_color[i].lower() == most_common:
                                final_color[i] = None

            # Sécurité finale : ne jamais baliser du noir standard
            for i in range(N_json):
                if is_black_color(final_color[i]): final_color[i] = None

        # 6. RECONSTRUCTION
        segments = []
        curr_seg = ""
        curr_b, curr_i, curr_c = False, False, None

        for i in range(N_json):
            b, it, c = final_bold[i], final_italic[i], final_color[i]
            if b != curr_b or it != curr_i or c != curr_c:
                if curr_seg: segments.append((curr_seg, curr_b, curr_i, curr_c))
                curr_seg = text[i]
                curr_b, curr_i, curr_c = b, it, c
            else:
                curr_seg += text[i]
        if curr_seg: segments.append((curr_seg, curr_b, curr_i, curr_c))

        result = ""
        for seg_txt, b, it, c in segments:
            if not seg_txt.strip():
                result += seg_txt
                continue
            lspace = len(seg_txt) - len(seg_txt.lstrip())
            rspace = len(seg_txt) - len(seg_txt.rstrip())
            core = seg_txt.strip()

            formatted = core
            if b: formatted = f"\\bf{{{formatted}}}"
            if it: formatted = f"\\it{{{formatted}}}"
            if c: formatted = f"\\color{{\"{formatted}\", {c}}}"
            result += seg_txt[:lspace] + formatted + (seg_txt[len(seg_txt) - rspace:] if rspace > 0 else "")

        node['parent'][node['key']] = result

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Fichier stylisé généré : {output_path}")


def main():
    # === NOUVEAUX DOSSIERS ===
    base_dir = Path(__file__).resolve().parent

    csv_dir = os.path.join(base_dir, "files_style")
    json_dir = os.path.join(base_dir, "extractionOut")
    output_dir = os.path.join(base_dir, "extractionOutStyle")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Trouve tous les JSON page_*.json
    json_files = glob.glob(os.path.join(json_dir, "page_*.json"))

    print(f"Fichiers trouvés : {len(json_files)}")

    for json_path in json_files:
        filename = os.path.basename(json_path)
        page_name = os.path.splitext(filename)[0]

        # CSV correspondant dans files_style
        csv_path = os.path.join(csv_dir, f"{page_name}.csv")

        if os.path.exists(csv_path):
            print(f"--> Traitement de {page_name}...")
            process_page(
                json_path,
                csv_path,
                os.path.join(output_dir, f"{page_name}--style.json")
            )
        else:
            print(f"[SKIP] {page_name} : CSV manquant.")


if __name__ == "__main__":
    main()