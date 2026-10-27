import cv2
import numpy as np
import streamlit as st


def crop_art(art, start_pct, end_pct):
  """Recorta a fatia horizontal exata da arte."""
  h, w = art.shape[:2]
  x1 = int(w * start_pct)
  x2 = int(w * end_pct)
  return art[:, x1:x2]


def apply_cylinder_mockup(
    bg, fg, pts_quad, curve_top_factor=0.0, curve_bot_factor=0.0, grid_steps=25
):
  """Reescreve a projeção cilíndrica mapeando a imagem através de uma malha vetorial exata."""
  bh, bw = bg.shape[:2]
  fh, fw = fg.shape[:2]

  TL, TR, BR, BL = (
      pts_quad[0],
      pts_quad[1],
      pts_quad[2],
      pts_quad[3],
  )

  # Construção da malha tridimensional
  src_grid = np.zeros((grid_steps + 1, grid_steps + 1, 2), dtype=np.float32)
  dst_grid = np.zeros((grid_steps + 1, grid_steps + 1, 2), dtype=np.float32)

  # Vetor de altura para projeção de curvatura
  vec_h = BL - TL
  h_length = np.linalg.norm(vec_h) + 1e-5
  unit_h = vec_h / h_length

  for j in range(grid_steps + 1):
    v = j / grid_steps
    for i in range(grid_steps + 1):
      u_linear = i / grid_steps

      # Compressão cilíndrica visual (efeito de rotação do cilindro)
      u_cyl = (1.0 - np.cos(u_linear * np.pi)) / 2.0

      # Ponto linear projetado no plano
      top_p = (1.0 - u_cyl) * TL + u_cyl * TR
      bot_p = (1.0 - u_cyl) * BL + u_cyl * BR
      p_base = (1.0 - v) * top_p + v * bot_p

      # Adiciona o arqueamento da boca e da base
      arc_t = np.sin(u_linear * np.pi) * curve_top_factor * h_length
      arc_b = np.sin(u_linear * np.pi) * curve_bot_factor * h_length
      arc_offset = (1.0 - v) * arc_t + v * arc_b

      p_final = p_base + unit_h * arc_offset

      dst_grid[j, i] = p_final
      src_grid[j, i] = [u_linear * fw, v * fh]

  result = bg.copy()

  # Renderização por interpolação de triângulos na malha
  for j in range(grid_steps):
    for i in range(grid_steps):
      sp0, sp1 = src_grid[j, i], src_grid[j, i + 1]
      sp2, sp3 = src_grid[j + 1, i], src_grid[j + 1, i + 1]

      dp0, dp1 = dst_grid[j, i], dst_grid[j, i + 1]
      dp2, dp3 = dst_grid[j + 1, i], dst_grid[j + 1, i + 1]

      # Triângulo 1
      draw_warped_triangle(fg, result, [sp0, sp1, sp2], [dp0, dp1, dp2])
      # Triângulo 2
      draw_warped_triangle(fg, result, [sp1, sp3, sp2], [dp1, dp3, dp2])

  return result


def draw_warped_triangle(src_img, dst_img, tri_src, tri_dst):
  """Aplica a transformação afim de cada célula da malha para o fundo."""
  tri_src = np.float32(tri_src)
  tri_dst = np.float32(tri_dst)

  r_dst = cv2.boundingRect(tri_dst)
  x, y, w, h = r_dst

  if (
      x < 0
      or y < 0
      or x + w > dst_img.shape[1]
      or y + h > dst_img.shape[0]
      or w <= 0
      or h <= 0
  ):
    return

  tri_dst_sub = tri_dst.copy()
  tri_dst_sub[:, 0] -= x
  tri_dst_sub[:, 1] -= y

  mat = cv2.getAffineTransform(tri_src, tri_dst_sub)
  warped = cv2.warpAffine(
      src_img,
      mat,
      (w, h),
      flags=cv2.INTER_LINEAR,
      borderMode=cv2.BORDER_REPLICATE,
  )

  mask = np.zeros((h, w), dtype=np.uint8)
  cv2.fillConvexPoly(
      mask, np.int32(np.round(tri_dst_sub)), 255, lineType=cv2.LINE_8
  )
  mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)

  roi = dst_img[y : y + h, x : x + w]
  mask_inv = cv2.bitwise_not(mask)
  bg_part = cv2.bitwise_and(roi, roi, mask=mask_inv)
  fg_part = cv2.bitwise_and(warped, warped, mask=mask)
  dst_img[y : y + h, x : x + w] = cv2.add(bg_part, fg_part)


# Configuração do Streamlit
st.set_page_config(page_title="Gerador de Mockup de Caneca", layout="centered")

st.title("☕ Gerador de Mockup de Caneca 3D")
st.write(
    "Envie a arte (20,5 x 10 cm), insira o nome do cliente e baixe seu mockup"
    " pronto!"
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
      with st.spinner("Reconstruindo malha 3D e renderizando..."):
        bh, bw = base_image.shape[:2]

        # COORDENADAS REESCRITAS (PRECISE ALIGNMENT)
        # 1. Topo Plano
        pts_top = np.array(
            [
                [bw * 0.038, bh * 0.145],
                [bw * 0.565, bh * 0.025],
                [bw * 0.612, bh * 0.300],
                [bw * 0.082, bh * 0.435],
            ],
            dtype=np.float32,
        )

        # 2. Caneca Esquerda (Encaixe cônico perfeito dentro das bordas da cerâmica)
        pts_left = np.array(
            [
                [bw * 0.075, bh * 0.605],
                [bw * 0.315, bh * 0.435],
                [bw * 0.420, bh * 0.785],
                [bw * 0.185, bh * 0.945],
            ],
            dtype=np.float32,
        )

        # 3. Caneca Centro
        pts_center = np.array(
            [
                [bw * 0.378, bh * 0.428],
                [bw * 0.628, bh * 0.428],
                [bw * 0.628, bh * 0.812],
                [bw * 0.378, bh * 0.812],
            ],
            dtype=np.float32,
        )

        # 4. Caneca Direita
        pts_right = np.array(
            [
                [bw * 0.665, bh * 0.488],
                [bw * 0.915, bh * 0.488],
                [bw * 0.915, bh * 0.885],
                [bw * 0.665, bh * 0.885],
            ],
            dtype=np.float32,
        )

        # RENDERIZAÇÃO DAS CAMADAS

        # Topo
        result = apply_cylinder_mockup(
            base_image, art, pts_top, curve_top_factor=0.0, curve_bot_factor=0.0
        )

        # Caneca Direita
        art_r = crop_art(art, 0.52, 1.0)
        result = apply_cylinder_mockup(
            result, art_r, pts_right, curve_top_factor=0.02, curve_bot_factor=0.02
        )

        # Caneca Centro
        art_c = crop_art(art, 0.25, 0.75)
        result = apply_cylinder_mockup(
            result,
            art_c,
            pts_center,
            curve_top_factor=0.015,
            curve_bot_factor=0.015,
        )

        # Caneca Esquerda (Curvatura profunda positiva para acompanhar o arco da boca sem vazar)
        art_l = crop_art(art, 0.0, 0.48)
        result = apply_cylinder_mockup(
            result,
            art_l,
            pts_left,
            curve_top_factor=0.085,
            curve_bot_factor=-0.035,
        )

        # Adição do Nome do Cliente
        if client_name:
          font = cv2.FONT_HERSHEY_SIMPLEX
          font_scale = bh * 0.001
          thickness = max(1, int(font_scale * 2))
          position = (int(bw * 0.02), int(bh * 0.95))
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
            result_rgb, caption="Seu mockup está pronto!", use_container_width=True
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
