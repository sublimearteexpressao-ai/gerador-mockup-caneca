import cv2
import numpy as np
import streamlit as st

def apply_perspective_crop(bg, fg, pts_dst, crop_x_start=0.0, crop_x_end=1.0):
    """Aplica um recorte da arte na região de destino do fundo."""
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

                # AS COORDENADAS ORIGINAIS DA SKILL RESTAURADAS
                pts_top = np.array([[bw*0.06, bh*0.15], [bw*0.56, bh*0.04], [bw*0.62, bh*0.30], [bw*0.08, bh*0.43]], dtype=float)
                pts_left = np.array([[bw*0.07, bh*0.60], [bw*0.33, bh*0.45], [bw*0.42, bh*0.81], [bw*0.12, bh*0.96]], dtype=float)
                pts_center = np.array([[bw*0.39, bh*0.43], [bw*0.64, bh*0.43], [bw*0.64, bh*0.81], [bw*0.39, bh*0.81]], dtype=float)
                pts_right = np.array([[bw*0.68, bh*0.48], [bw*0.90, bh*0.48], [bw*0.90, bh*0.88], [bw*0.68, bh*0.88]], dtype=float)

                # Aplicando os recortes nas coordenadas corretas
                result = apply_perspective_crop(base_image, art, pts_top, 0.0, 1.0)
                result = apply_perspective_crop(result, art, pts_left, 0.0, 0.45)
                result = apply_perspective_crop(result, art, pts_center, 0.25, 0.75)
                result = apply_perspective_crop(result, art, pts_right, 0.55, 1.0)

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
