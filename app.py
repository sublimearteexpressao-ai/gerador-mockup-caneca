import cv2
import numpy as np
import streamlit as st

def warp_triangle(img_src, img_dst, tri_src, tri_dst):
    """Deforma triângulos individuais para criar a malha abulada contínua."""
    r_dst = cv2.boundingRect(tri_dst)
    x, y, w, h = r_dst
    if x < 0 or y < 0 or x + w > img_dst.shape[1] or y + h > img_dst.shape[0] or w <= 0 or h <= 0:
        return

    tri_dst_cropped = tri_dst.copy()
    tri_dst_cropped[:, 0] -= x
    tri_dst_cropped[:, 1] -= y

    mat = cv2.getAffineTransform(tri_src, tri_dst_cropped)
    warped = cv2.warpAffine(img_src, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(np.round(tri_dst_cropped)), 255, lineType=cv2.LINE_8)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    roi = img_dst[y:y+h, x:x+w]
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img_fg = cv2.bitwise_and(warped, warped, mask=mask)
    img_dst[y:y+h, x:x+w] = cv2.add(img_bg, img_fg)

def apply_mesh_curved_perspective(bg, fg, pts_dst, arc_top=0.0, arc_bottom=0.0, grid_size=16):
    """
    Deforma a PRÓPRIA ARTE em uma malha curva (abulamento real da imagem).
    """
    h_fg, w_fg = fg.shape[:2]
    TL, TR, BR, BL = pts_dst[0], pts_dst[1], pts_dst[2], pts_dst[3]

    dst_pts = np.zeros((grid_size + 1, grid_size + 1, 2), dtype=np.float32)
    src_pts = np.zeros((grid_size + 1, grid_size + 1, 2), dtype=np.float32)

    h_ref = np.linalg.norm(BL - TL)

    for j in range(grid_size + 1):
        v = j / grid_size
        for i in range(grid_size + 1):
            u = i / grid_size
            
            # Posição linear base na perspectiva
            top_p = (1.0 - u) * TL + u * TR
            bot_p = (1.0 - u) * BL + u * BR
            p_base = (1.0 - v) * top_p + v * bot_p

            # Aplica a curvatura abulada na arte de acordo com a posição horizontal (u)
            arc_factor_top = np.sin(u * np.pi) * arc_top * h_ref
            arc_factor_bot = np.sin(u * np.pi) * arc_bottom * h_ref
            
            # Interpola o abulamento entre o topo e a base
            arc_offset = (1.0 - v) * arc_factor_top + v * arc_factor_bot
            
            dst_pts[j, i] = [p_base[0], p_base[1] + arc_offset]
            src_pts[j, i] = [u * w_fg, v * h_fg]

    result = bg.copy()
    
    # Renderiza a malha 3D sobre o fundo
    for j in range(grid_size):
        for i in range(grid_size):
            sp0, sp1, sp2, sp3 = src_pts[j, i], src_pts[j, i+1], src_pts[j+1, i], src_pts[j+1, i+1]
            dp0, dp1, dp2, dp3 = dst_pts[j, i], dst_pts[j, i+1], dst_pts[j+1, i], dst_pts[j+1, i+1]

            tri_src1, tri_dst1 = np.float32([sp0, sp1, sp2]), np.float32([dp0, dp1, dp2])
            warp_triangle(fg, result, tri_src1, tri_dst1)

            tri_src2, tri_dst2 = np.float32([sp1, sp3, sp2]), np.float32([dp1, dp3, dp2])
            warp_triangle(fg, result, tri_src2, tri_dst2)

    return result

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
            with st.spinner("Deformando arte e gerando curvatura 3D..."):
                bh, bw = base_image.shape[:2]

                # 1. Topo Plano (Inclinado)
                pts_top = np.array([
                    [bw*0.038, bh*0.145], [bw*0.565, bh*0.025], 
                    [bw*0.612, bh*0.300], [bw*0.082, bh*0.435]
                ], dtype=float)
                
                # 2. Caneca Esquerda
                pts_left = np.array([
                    [bw*0.068, bh*0.595], [bw*0.312, bh*0.425], 
                    [bw*0.418, bh*0.795], [bw*0.180, bh*0.935]
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

                # APLICAÇÃO DA DEFORMAÇÃO NA PRÓPRIA IMAGEM

                # 1. Topo Plano
                result = apply_mesh_curved_perspective(base_image, art, pts_top, arc_top=0.0, arc_bottom=0.0)

                # 2. Caneca Direita (Abulamento leve)
                art_right = crop_art(art, 0.52, 1.0)
                result = apply_mesh_curved_perspective(result, art_right, pts_right, arc_top=0.02, arc_bottom=0.02)

                # 3. Caneca Centro (Abulamento leve)
                art_center = crop_art(art, 0.25, 0.75)
                result = apply_mesh_curved_perspective(result, art_center, pts_center, arc_top=0.015, arc_bottom=0.015)

                # 4. Caneca Esquerda (ABULAMENTO PROFUNDO NA PRÓPRIA ARTE: arc_top positivo puxa os pixels para baixo)
                art_left = crop_art(art, 0.0, 0.48)
                result = apply_mesh_curved_perspective(result, art_left, pts_left, arc_top=0.100, arc_bottom=0.040)

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
