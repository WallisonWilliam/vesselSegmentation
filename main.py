import numpy as np
from scipy.signal import convolve2d
from PIL import Image
from skimage import filters, morphology


def mean_filter(input_image, r):
    """Aplica um filtro de média na imagem de entrada."""
    kernel = np.ones((r, r), dtype=np.float32) / (r * r)
    # Assegurar que o filtro seja aplicado em cada canal de cor
    if input_image.ndim == 3:
        filtered_image = np.zeros_like(input_image)
        for i in range(input_image.shape[2]):
            filtered_image[:, :, i] = convolve2d(input_image[:, :, i], kernel, boundary='symm', mode='same')
        return filtered_image
    else:
        return convolve2d(input_image, kernel, boundary='symm', mode='same')


def guided_filter(F, G, r, eps):
    """Aplica a filtragem guiada em um mapa de características com uma imagem de orientação."""
    F_mean = mean_filter(F, r)
    G_mean = mean_filter(G, r)
    G2_mean = mean_filter(G * G, r)
    GF_mean = mean_filter(F * G, r)

    # Cálculo da Variância e Covariância
    G_var = G2_mean - G_mean * G_mean
    GF_cov = GF_mean - G_mean * F_mean

    # Estimação dos Coeficientes Lineares a e b
    a = GF_cov / (G_var + eps)
    b = F_mean - a * G_mean

    # Suavização dos Coeficientes a e b
    a_mean = mean_filter(a, r)
    b_mean = mean_filter(b, r)

    # Geração do Mapa de Saída O
    O = a_mean * F + b_mean

    # Adiciona uma dimensão de canal se F e O forem 2D
    if F.ndim == 2:
        F = F[:, :, np.newaxis]
    if O.ndim == 2:
        O = O[:, :, np.newaxis]

    # Concatena F e O ao longo do eixo do canal
    FO_concat = np.concatenate((F, O), axis=2)

    # Aplica a convolução
    F_output = convolution(FO_concat, 3, 64)  # Exemplo: kernel 3x3, 64 filtros
    return F_output


def convolution(input_concat, kernel_size, num_filters):
    height, width = input_concat.shape[:2]
    output = np.zeros((height, width, num_filters))

    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)

    for i in range(num_filters):
        for channel in range(input_concat.shape[2]):
            output[:, :, i] += convolve2d(input_concat[:, :, channel], kernel, mode='same')

    return output


def load_image(image_path):
    """ Carrega uma imagem colorida e a converte em um array de ponto flutuante. """
    with Image.open(image_path) as img:
        return np.asarray(img, dtype=np.float32) / 255


def save_image(image, output_path):
    """ Salva um array NumPy como uma imagem.

    Args:
    - image: Imagem para ser salva (array NumPy).
    - output_path: Caminho do arquivo onde a imagem será salva.
    """
    # Se a imagem for 3D (colorida), converter para escala de cinza antes de salvar.
    if image.ndim == 3 and image.shape[2] == 3:
        image = np.mean(image, axis=2)

    # Se a imagem já for 2D (escala de cinza ou binária), pode salvar diretamente.
    image_to_save = Image.fromarray(np.uint8(image * 255))
    image_to_save.save(output_path)


def segment_vessels(F_filtered):
    """Segmenta os vasos sanguíneos de uma imagem."""
    # Converter para escala de cinza
    F_gray = np.mean(F_filtered, axis=2)

    # Aplicar um filtro de bordas, como Sobel, para realçar os vasos sanguíneos
    edges = filters.sobel(F_gray)

    # Limiarização utilizando o método de Otsu para binarizar a imagem
    thresh = filters.threshold_otsu(edges)
    binary_image = edges > thresh

    # Remover pequenos objetos (ruído)
    cleaned_image = morphology.remove_small_objects(binary_image, min_size=30)

    # Dilatar os vasos sanguíneos para torná-los mais visíveis
    selem = morphology.disk(1)
    dilated_image = morphology.dilation(cleaned_image, selem)

    return dilated_image


image_path = '1.png'
F = load_image(image_path)
G = F.copy()  # A imagem de orientação

r = 2  # Raio do filtro de média
eps = 1e-8  # Termo de regularização

# Aplicar o filtro guiado
F_filtered = guided_filter(F, G, r, eps)

# Segmentar os vasos sanguíneos
F_segmented = segment_vessels(F_filtered)

# Salvar a imagem segmentada
output_path = 'C:\\Users\\Wallison\\Desktop\\pythonProject2\\2.png'
save_image(F_segmented, output_path)
