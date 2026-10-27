import cv2
import numpy as np
import streamlit as st

def apply_precise_curved_perspective(bg, fg, pts_dst, arc_top=0.0, arc_bottom=0.0):
    """Aplica a perspectiva e recorta com máscara curvada exatamente no formato da cerâmica."""
    h_fg, w_fg = fg.shape[:2]
    pts_src = np.array([[0, 0], [w_fg, 0], [w_fg, h_fg], [0, h_fg]], dtype=float)
    
    h_matrix, _ = cv2.findHomography(pts_src, pts_dst)
    warped = cv2.warpPerspective(fg, h_matrix, (bg.shape[1], bg.shape[0]))
    
    mask = np.zeros((bg.shape[0], bg.shape[1]), dtype=np.uint8)
    
    n_pts = 40
    curve_points = []
    
    # Curva Superior (Topo da Caneca)
    tl, tr = pts_dst[0], pts_dst[1]
    for i in range(n_pts + 1):
        t = i / n_pts
        p = (1 - t) * tl + t * tr
        arc = np.sin(t * np.pi) * (pts_dst[3][1] - pts_dst[0][1]) * arc_top
        curve_points.append([p[0], p[1] + arc])
        
    # Curva Inferior (Base da Caneca)
    br, bl = pts_dst[2], pts_dst[3]
    for i in range(n_pts + 1):
        t = i / n_pts
        p = (1 - t) * br + t * bl
        arc = np.sin(t * np.pi) * (pts_dst[2][1] - pts_dst[1][1]) * arc_bottom
        curve_points.append([p[0], p[1] + arc])
        
    curved_poly = np.array(curve_points, dtype=np.int32)
    cv2.fillConvexPoly(mask, curved_poly, 255)
    
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    bg_bg = cv2.bitwise_and(bg, bg, mask=cv2.bitwise_not(mask))
    fg_fg = cv2.bitwise_and(warped, warped, mask=mask)
    return cv2.add(bg_bg, fg_fg)

def crop_art(art, start_pct, end_pct):
    """Recorta a fatia exata da arte."""
    h, w = art.shape[:2]
    x1 = int(w * start_pct)
    x2 = int(w * end_pct)
    return art[:, x1:x2]

st.set_page_config(page_title="Gerador de Mockup de Caneca", layout="centered")

st.title("☕ Gerador de Mockup de Caneca 3D")
st.write("Envie a arte (20,5 x 10 cm), insira o nome do cliente e baixe seu mockup pronto!")

base_image = cv2.imread("Base_em_branco.jpg")

if base_image is None:
    st.error("Erro: A imagem 'Base_em_branco.jpg' não foi encontrada no repositório.")
else:
    uploaded_file = st.file_uploader("Selecione a arte (imagem JPG ou PNG)", type=["jpg", "jpeg", "png"])
    client_name = st.text_input("Nome do Cliente (opcional)")

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        art = cv2.imdecode(file_bytes, 1)

        if st.button("Gerar Mockup 🚀"):
            with st.spinner("Gerando mockup preciso..."):
                bh, bw = base_image.shape[:2]

                # 1. Topo Plano (Inclinado)
                pts_top = np.array([
                    [bw*0.038, bh*0.145], [bw*0.565, bh*0.025], 
                    [bw*0.612, bh*0.300], [bw*0.082, bh*0.435]
                ], dtype=float)
                
                # 2. Caneca Esquerda REFINADA (Encurtada na base e alinhada ao topo da cerâmica)
                pts_left = np.array([
                    [bw*0.085, bh*0.600], [bw*0.315, bh*0.460], 
                    [bw*0.410, bh*0.785], [bw*0.185, bh*0.925]
                ], dtype=float)
                
                # 3. Caneca Centro
                pts_center = np.array([
                    [bw*0.378, bh*0.428], [bw*0.628, bh*0.428], 
                    [bw*0.628, bh*0.812], [bw*0.378, bh*0.812]
                ], dtype=float)
                
                # 4. Caneca Direita
                pts_right = np.array([
                    [bw*0.665, bh*0.488], [bw*0.915, bh*0.488], 
                    [bw*0.915, bh*0.885], [bw*0.665, bh*0.885]
                ], dtype=float)

                # APLICAÇÃO DAS CAMADAS E MÁSCARAS
                
                # 1. Visão Superior Plana
                result = apply_precise_curved_perspective(base_image, art, pts_top, arc_top=0.0, arc_bottom=0.0)

                # 2. Caneca Direita (Fatia 52% a 100%)
                art_right = crop_art(art, 0.52, 1.0)
                result = apply_precise_curved_perspective(result, art_right, pts_right, arc_top=0.015, arc_bottom=0.015)

                # 3. Caneca Centro (Fatia 25% a 75%)
                art_center = crop_art(art, 0.25, 0.75)
                result = apply_precise_curved_perspective(result, art_center, pts_center, arc_top=0.012, arc_bottom=0.012)

                # 4. Caneca Esquerda REFINADA (Fatia 0% a 48% - Arc_top ajustado para -0.025)
                art_left = crop_art(art, 0.0, 0.48)
                result = apply_precise_curved_perspective(result, art_left, pts_left, arc_top=-0.025, arc_bottom=0.020)

                # Nome do Cliente
                if client_name:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = bh * 0.001
                    thickness = max(1, int(font_scale * 2))
                    position = (int(bw * 0.02), int(bh * 0.95))
                    text = f"Cliente: {client_name}"
                    cv2.putText(result, text, position, font, font_scale, (255, 255, 255), thickness + 3, cv2.LINE_AA)
                    cv2.putText(result, text, position, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, caption="Seu mockup está pronto!", use_container_width=True)

                _, im_png = cv2.imencode(".png", result)
                st.download_button(
                    label="Baixar Mockup 📥",
                    data=im_png.tobytes(),
                    file_name=f"mockup_{client_name.replace(' ', '_') if client_name else 'caneca'}.png",
                    mime="image/png"
                )
