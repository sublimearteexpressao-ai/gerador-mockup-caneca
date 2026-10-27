import cv2
import numpy as np
import streamlit as st

def warp_triangle(img_src, img_dst, tri_src, tri_dst):
    """Deforma triângulos da malha mantendo os limites sem estourar as bordas."""
    r = cv2.boundingRect(tri_dst)
    x, y, w, h = r
    if x < 0 or y < 0 or x+w > img_dst.shape[1] or y+h > img_dst.shape[0] or w <= 0 or h <= 0:
        return
        
    tri_dst_cropped = tri_dst.copy()
    tri_dst_cropped[:, 0] -= x
    tri_dst_cropped[:, 1] -= y
    
    mat = cv2.getAffineTransform(tri_src, tri_dst_cropped)
    warped = cv2.warpAffine(img_src, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(np.round(tri_dst_cropped)), 255, lineType=cv2.LINE_8)
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    roi = img_dst[y:y+h, x:x+w]
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    img_fg = cv2.bitwise_and(warped, warped, mask=mask)
    img_dst[y:y+h, x:x+w] = cv2.add(img_bg, img_fg)

def apply_cylindrical_perspective(bg, fg, corners, crop_x_start=0.0, crop_x_end=1.0, arc_amount=0.025, grid_size=12):
    """
    Aplica a curvatura cilíndrica usufruindo EXATAMENTE das coordenadas de 4 pontos da skill original.
    """
    h_fg_full, w_fg_full = fg.shape[:2]
    x1 = int(crop_x_start * w_fg_full)
    x2 = int(crop_x_end * w_fg_full)
    fg_crop = fg[:, x1:x2]
    h_fg, w_fg = fg_crop.shape[:2]

    TL, TR, BR, BL = corners[0], corners[1], corners[2], corners[3]

    dst_pts = np.zeros((grid_size + 1, grid_size + 1, 2))
    src_pts = np.zeros((grid_size + 1, grid_size + 1, 2))

    height_vec = (BL + BR) / 2.0 - (TL + TR) / 2.0
    h_len = np.linalg.norm(height_vec) + 1e-6

    for j in range(grid_size + 1):
        v = j / grid_size
        for i in range(grid_size + 1):
            u_linear = i / grid_size
            # Compressão angular do cilindro nas bordas
            u_curve = (1.0 - np.cos(u_linear * np.pi)) / 2.0

            top_p = (1.0 - u_curve) * TL + u_curve * TR
            bot_p = (1.0 - u_curve) * BL + u_curve * BR
            p_base = (1.0 - v) * top_p + v * bot_p

            # Curvatura suave para eliminar linhas retas nas bordas superior e inferior
            arc = np.sin(u_linear * np.pi) * arc_amount * h_len
            p_curved = p_base + (height_vec / h_len) * arc

            dst_pts[j, i] = p_curved
            src_pts[j, i] = [u_linear * w_fg, v * h_fg]

    result = bg.copy()
    for j in range(grid_size):
        for i in range(grid_size):
            sp0, sp1, sp2, sp3 = src_pts[j,i], src_pts[j,i+1], src_pts[j+1,i], src_pts[j+1,i+1]
            dp0, dp1, dp2, dp3 = dst_pts[j,i], dst_pts[j,i+1], dst_pts[j+1,i], dst_pts[j+1,i+1]
            
            tri_src1, tri_dst1 = np.float32([sp0, sp1, sp2]), np.float32([dp0, dp1, dp2])
            warp_triangle(fg_crop, result, tri_src1, tri_dst1)
            
            tri_src2, tri_dst2 = np.float32([sp1, sp3, sp2]), np.float32([dp1, dp3, dp2])
            warp_triangle(fg_crop, result, tri_src2, tri_dst2)

    return result

def apply_perspective_crop(bg, fg, pts_dst, crop_x_start=0.0, crop_x_end=1.0):
    """Aplicado apenas na folha plana do topo."""
    h_fg, w_fg = fg.shape[:2]
    x1 = int(crop_x_start * w_fg)
    x2 = int(crop_x_end * w_fg)
    fg_cropped = fg[:, x1:x2]

    h_c, w_c = fg_cropped.shape[:2]
    pts_src = np.array([[0, 0], [w_c, 0], [w_c, h_c], [0, h_c]], dtype=float)

    h_matrix, _ = cv2.findHomography(pts_src, pts_dst)
    warped = cv2.warpPerspective(fg_cropped, h_matrix, (bg.shape[1], bg.shape[0]))

    mask = np.zeros((bg.shape[0], bg.shape[1]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts_dst.astype(int), 255)

    bg_bg = cv2.bitwise_and(bg, bg, mask=cv2.bitwise_not(mask))
    fg_fg = cv2.bitwise_and(warped, warped, mask=mask)
    return cv2.add(bg_bg, fg_fg)


st.set_page_config(page_title="Gerador de Mockup de Caneca", layout="centered")

st.title("☕ Gerador de Mockup de Caneca")
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
            with st.spinner("Processando..."):
                bh, bw = base_image.shape[:2]

                # COORDENADAS ORIGINAIS DA SKILL RESTAURADAS
                pts_top = np.array([[bw*0.06, bh*0.15], [bw*0.56, bh*0.04], [bw*0.62, bh*0.30], [bw*0.08, bh*0.43]], dtype=float)
                pts_left = np.array([[bw*0.07, bh*0.60], [bw*0.33, bh*0.45], [bw*0.42, bh*0.81], [bw*0.12, bh*0.96]], dtype=float)
                pts_center = np.array([[bw*0.39, bh*0.43], [bw*0.64, bh*0.43], [bw*0.64, bh*0.81], [bw*0.39, bh*0.81]], dtype=float)
                pts_right = np.array([[bw*0.68, bh*0.48], [bw*0.90, bh*0.48], [bw*0.90, bh*0.88], [bw*0.68, bh*0.88]], dtype=float)

                # 1. Visão Superior Plana
                result = apply_perspective_crop(base_image, art, pts_top, 0.0, 1.0)
                
                # 2. Caneca Direita (Lado direito da arte)
                result = apply_cylindrical_perspective(result, art, pts_right, 0.55, 1.0, arc_amount=0.02)
                
                # 3. Caneca Centro (Centro da arte)
                result = apply_cylindrical_perspective(result, art, pts_center, 0.25, 0.75, arc_amount=0.02)
                
                # 4. Caneca Esquerda (Lado esquerdo da arte - sobrepõe a do centro)
                result = apply_cylindrical_perspective(result, art, pts_left, 0.0, 0.45, arc_amount=0.02)

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
