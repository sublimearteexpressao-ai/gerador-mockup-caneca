import cv2
import numpy as np
import streamlit as st


def apply_perspective_crop(bg, fg, pts_dst, crop_x_start=0.0, crop_x_end=1.0):
  """Aplica uma secção/recorte da arte na região de destino do fundo."""
  h_fg, w_fg = fg.shape[:2]

  # Recorta apenas a faixa horizontal correspondente da arte
  x1 = int(crop_x_start * w_fg)
  x2 = int(crop_x_end * w_fg)
  fg_cropped = fg[:, x1:x2]

  h_c, w_c = fg_cropped.shape[:2]
  pts_src = np.array([[0, 0], [w_c, 0], [w_c, h_c], [0, h_c]], dtype=float)

  h_matrix, _ = cv2.findHomography(pts_src, pts_dst)
  warped = cv2.warpPerspective(
      fg_cropped, h_matrix, (bg.shape[1], bg.shape[0])
  )

  mask = np.zeros((bg.shape[0], bg.shape[1]), dtype=np.uint8)
  cv2.fillConvexPoly(mask, pts_dst.astype(int), 255)

  bg_bg = cv2.bitwise_and(bg, bg, mask=cv2.bitwise_not(mask))
  fg_fg = cv2.bitwise_and(warped, warped, mask=mask)
  return cv2.add(bg_bg, fg_fg)


st.set_page_config(page_title="Gerador de Mockup de Caneca", layout="centered")

st.title("☕ Gerador de Mockup de Caneca")
st.write(
    "Envie a arte (20,5 x 10 cm), insira o nome do cliente e descarregue o seu"
    " mockup pronto!"
)

base_image = cv2.imread("Base_em_branco.jpg")

if base_image is None:
  st.error(
      "Erro: A imagem 'Base_em_branco.jpg' não foi encontrada no repositório."
  )
else:
  uploaded_file = st.file_uploader(
      "Selecione a arte (imagem JPG ou PNG)", type=["jpg", "jpeg", "png"]
  )
  client_name = st.text_input("Nome do Cliente (opcional)")

  if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    art = cv2.imdecode(file_bytes, 1)

    if st.button("Gerar Mockup 🚀"):
      with st.spinner("A processar..."):
        bh, bw = base_image.shape[:2]

        # Coordenadas ajustadas das canecas
        pts_top = np.array(
            [
                [bw * 0.05, bh * 0.15],
                [bw * 0.56, bh * 0.04],
                [bw * 0.61, bh * 0.30],
                [bw * 0.08, bh * 0.43],
            ],
            dtype=float,
        )

        pts_left = np.array(
            [
                [bw * 0.06, bh * 0.58],
                [bw * 0.35, bh * 0.43],
                [bw * 0.41, bh * 0.81],
                [bw * 0.12, bh * 0.96],
            ],
            dtype=float,
        )

        pts_center = np.array(
            [
                [bw * 0.38, bh * 0.42],
                [bw * 0.62, bh * 0.42],
                [bw * 0.62, bh * 0.81],
                [bw * 0.38, bh * 0.81],
            ],
            dtype=float,
        )

        pts_right = np.array(
            [
                [bw * 0.67, bh * 0.49],
                [bw * 0.91, bh * 0.49],
                [bw * 0.91, bh * 0.88],
                [bw * 0.67, bh * 0.88],
            ],
            dtype=float,
        )

        # 1. Visão plana no topo (usa a arte inteira: 0.0 a 1.0)
        result = apply_perspective_crop(
            base_image, art, pts_top, crop_x_start=0.0, crop_x_end=1.0
        )

        # 2. Caneca Esquerda (usa a secção inicial/esquerda: 0.0 a 0.55)
        result = apply_perspective_crop(
            result, art, pts_left, crop_x_start=0.0, crop_x_end=0.55
        )

        # 3. Caneca Centro (usa a secção central: 0.20 a 0.80)
        result = apply_perspective_crop(
            result, art, pts_center, crop_x_start=0.20, crop_x_end=0.80
        )

        # 4. Caneca Direita (usa a secção final/direita: 0.45 a 1.0)
        result = apply_perspective_crop(
            result, art, pts_right, crop_x_start=0.45, crop_x_end=1.0
        )

        # Inserção do Nome do Cliente
        if client_name:
          font = cv2.FONT_HERSHEY_SIMPLEX
          font_scale = bh * 0.001
          thickness = max(1, int(font_scale * 2))
          position = (int(bw * 0.04), int(bh * 0.94))
          text = f"Cliente: {client_name}"
          cv2.putText(
              result,
              text,
              position,
              font,
              font_scale,
              (255, 255, 255),
              thickness + 3,
              cv2.LINE_AA,
          )
          cv2.putText(
              result,
              text,
              position,
              font,
              font_scale,
              (0, 0, 0),
              thickness,
              cv2.LINE_AA,
          )

        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        st.image(
            result_rgb, caption="O seu mockup está pronto!", use_container_width=True
        )

        _, im_png = cv2.imencode(".png", result)
        st.download_button(
            label="Baixar Mockup 📥",
            data=im_png.tobytes(),
            file_name=(
                f"mockup_{client_name.replace(' ', '_') if client_name else 'caneca'}.png"
            ),
            mime="image/png",
        )
