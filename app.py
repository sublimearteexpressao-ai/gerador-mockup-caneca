import cv2
import numpy as np
import streamlit as st

def apply_perspective(bg, fg, pts_dst):
    """Aplica a perspectiva exatamente como na versão original da skill."""
    h, w = fg.shape[:2]
    pts_src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    h_matrix, _ = cv2.findHomography(pts_src, pts_dst)
    warped = cv2.warpPerspective(fg, h_matrix, (bg.shape[1], bg.shape[0]))
    
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

                # Coordenadas originais exatas da skill
                pts_top = np.array([[bw*0.06, bh*0.15], [bw*0.56, bh*0.04], [bw*0.62, bh*0.30], [bw*0.08, bh*0.43]], dtype=float)
                pts_left = np.array([[bw*0.07, bh*0.60], [bw*0.33, bh*0.45], [bw*0.42, bh*0.81], [bw*0.12, bh*0.96]], dtype=float)
                pts_center = np.array([[bw*0.39, bh*0.43], [bw*0.64, bh*0.43], [bw*0.64, bh*0.81], [bw*0.39, bh*0.81]], dtype=float)
                pts_right = np.array([[bw*0.68, bh*0.48], [bw*0.90, bh*0.48], [bw*0.90, bh*0.88], [bw*0.68, bh*0.88]], dtype=float)

                # Aplica a arte nas 4 regiões idêntico ao script inicial
                result = apply_perspective(base_image, art, pts_top)
                result = apply_perspective(result, art, pts_left)
                result = apply_perspective(result, art, pts_center)
                result = apply_perspective(result, art, pts_right)

                # Inserção do Nome do Cliente
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
