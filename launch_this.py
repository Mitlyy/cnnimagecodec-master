from CNNImageCodec import *
from sklearn.cluster import KMeans

# Assuming EntropyEncoder and EntropyDecoder are defined elsewhere

def NeuralCompressor_min_max(enc, dec, b=2):


    encoded_layers = enc.predict(xtest, batch_size=NumImagesToShow)
    max_encoded_layers = np.zeros(NumImagesToShow, np.float16)
    min_encoded_layers = np.zeros(NumImagesToShow, np.float16)

    for i in range(0, NumImagesToShow):
        max_encoded_layers[i] = np.max(encoded_layers[i])
        min_encoded_layers[i] = np.min(encoded_layers[i])
        encoded_layers[i] = (encoded_layers[i] - min_encoded_layers[i]) / (max_encoded_layers[i] - min_encoded_layers[i])

    quantization_scale = pow(2, b)  # Scale based on the number of bits
    encoded_layers_quantized = np.clip(encoded_layers, 0, 0.9999999)
    encoded_layers_quantized = np.floor(encoded_layers_quantized * quantization_scale).astype(np.int32)

    bpp = np.zeros(NumImagesToShow, np.float16)
    declayers = np.zeros((NumImagesToShow, 16, 16, 16), np.uint8)
    for i in range(NumImagesToShow):
        binfilename = f'image{i}.bin'
        EntropyEncoder(binfilename, encoded_layers_quantized[i], 16, 16, 16)
        bytesize = os.path.getsize(binfilename)
        bpp[i] = bytesize * 8 / (w * h)  # Bits per pixel
        declayers[i] = EntropyDecoder(binfilename, 16, 16, 16)

    print("Bits per pixel:NeuralCompressor_min_max", bpp)
    shift = 1.0 / pow(2, b + 1)
    declayers = declayers.astype(np.float32) / pow(2, b)
    declayers = declayers + shift
    decoded_layers = np.zeros((NumImagesToShow, 16, 16, 16), np.float32)

    for i in range(NumImagesToShow):
        decoded_layers[i] = (declayers[i] * (max_encoded_layers[i] - min_encoded_layers[i])) + min_encoded_layers[i]

    decoded_imgs = dec.predict(decoded_layers, batch_size=NumImagesToShow)
    decoded_imgsQ = dec.predict(decoded_layers, batch_size=NumImagesToShow)  # Quantized decoding

    return bpp, decoded_imgs, decoded_imgsQ


def NeuralCompressor_vector(enc, dec, b=2):
    # Получение закодированных слоев
    encoded_layers = enc.predict(xtest, batch_size=NumImagesToShow)

    # Преобразование слоев в векторы для кластеризации
    flat_encoded_layers = encoded_layers.reshape(len(xtest), -1)

    # Применение векторного квантования с использованием KMeans
    kmeans = KMeans(n_clusters=2**b, random_state=42)
    quantized_labels = []
    centroids = []

    for i in range(NumImagesToShow):
        kmeans.fit(flat_encoded_layers[i].reshape(-1, 1))
        quantized_labels.append(kmeans.labels_)
        centroids.append(kmeans.cluster_centers_)

    quantized_labels = [labels.reshape(encoded_layers.shape[1:]) for labels in quantized_labels]

    # Сохранение квантованных данных и кодирование
    bpp = np.zeros(NumImagesToShow, np.float16)
    declayers = np.zeros_like(encoded_layers, dtype=np.float32)

    for i in range(NumImagesToShow):
        binfilename = f'image{i}.bin'
        EntropyEncoder(binfilename, quantized_labels[i], 16, 16, 16)
        bytesize = os.path.getsize(binfilename)
        bpp[i] = bytesize * 8 / (w * h)  # Биты на пиксель
        declayers[i] = EntropyDecoder(binfilename, 16, 16, 16)

    # Восстановление данных с использованием центроидов
    decoded_layers = np.zeros_like(encoded_layers)
    for i in range(NumImagesToShow):
        decoded_layers[i] = centroids[i][declayers[i].astype(int)].reshape(encoded_layers.shape[1:])

    # Декодирование изображений
    decoded_imgs = dec.predict(decoded_layers, batch_size=NumImagesToShow)
    decoded_imgsQ = dec.predict(decoded_layers, batch_size=NumImagesToShow)  # Квантованное декодирование

    return bpp, decoded_imgs, decoded_imgsQ




        
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(decoded_images):
    """
    Вычисление среднего SSIM для декодированных изображений.
    """
    ssim_value = (
        ssim(xtest[0, :, :, 0], decoded_images[0, :, :, 0], data_range=1.0) +
        ssim(xtest[0, :, :, 1], decoded_images[0, :, :, 1], data_range=1.0) +
        ssim(xtest[0, :, :, 2], decoded_images[0,  :, :, 2], data_range=1.0)
    ) / 3.0
    return np.mean(ssim_value)


def calculate_ssim_bpp(xtest, decoded_images, bpp):
    """
    Вычисление среднего SSIM/bpp для декодированных изображений.
    """
    ssim_bpp_values = []
    for i in range(len(xtest)):
        ssim_value = (
            ssim(xtest[i, :, :, 0], decoded_images[i, :, :, 0], data_range=1.0) +
            ssim(xtest[i, :, :, 1], decoded_images[i, :, :, 1], data_range=1.0) +
            ssim(xtest[i, :, :, 2], decoded_images[i, :, :, 2], data_range=1.0)
        ) / 3.0
        ssim_bpp_values.append(ssim_value / bpp[i])
    return np.mean(ssim_bpp_values)


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

def plot_ssim_bpp(testfolder, trainfolder, bees, NumImagesToShow=5):
    """
    Построение графика среднего SSIM/bpp для AE2, JPEG и предложенного кодека.
    """
    xtest = LoadImagesFromFolder(testfolder) / 255  


    encoder, decoder = ImageCodecModel(trainfolder, 0)
    encoder2, decoder2 = ImageCodecModel(trainfolder, 1)

    results = {'AE1_min_max': [], 'AE2_min_max': [], 'AE1_vector': [], 'AE2_vector': [],'AE1_old': [], 'AE2_old': [], 'JPEG': []}

    for b in bees:
        print("BEEEEE", b)
        bpp2, _, decoded_imgsQ2 = NeuralCompressor_min_max(encoder2, decoder2, b=b)
        avg_ssim_bpp_ae2 = calculate_ssim_bpp(xtest[:NumImagesToShow], decoded_imgsQ2, bpp2)
        results['AE2_min_max'].append((b, avg_ssim_bpp_ae2))

        bpp, _, decoded_imgsQ = NeuralCompressor_min_max(encoder, decoder, b=b)
        avg_ssim_bpp_ae = calculate_ssim_bpp(xtest[:NumImagesToShow], decoded_imgsQ, bpp)
        results['AE1_min_max'].append((b, avg_ssim_bpp_ae))


        jpeg_ssim_bpp = []
        for i in range(NumImagesToShow):
            _, JPEG_bpp, JPEG_ssim = JPEGRDSingleImage(xtest[i, :, :, :], b, i)
            jpeg_ssim_bpp.append(JPEG_ssim / JPEG_bpp)
        results['JPEG'].append((b, np.mean(jpeg_ssim_bpp)))


        bpp2, _, decoded_imgsQ2 = NeuralCompressor_vector(encoder2, decoder2, b=b)
        avg_ssim_bpp_ae2 = calculate_ssim_bpp(xtest[:NumImagesToShow], decoded_imgsQ2, bpp2)
        results['AE2_vector'].append((b, avg_ssim_bpp_ae2))

        bpp, _, decoded_imgsQ = NeuralCompressor_vector(encoder, decoder, b=b)
        avg_ssim_bpp_ae = calculate_ssim_bpp(xtest[:NumImagesToShow], decoded_imgsQ, bpp)
        results['AE1_vector'].append((b, avg_ssim_bpp_ae))
        
        bpp2, _, decoded_imgsQ2 = NeuralCompressor(encoder2, decoder2,xtest, b=b)
        avg_ssim_bpp_ae2 = calculate_ssim_bpp(xtest[:NumImagesToShow], decoded_imgsQ2, bpp2)
        results['AE2_old'].append((b, avg_ssim_bpp_ae2))

        bpp, _, decoded_imgsQ = NeuralCompressor(encoder, decoder,xtest, b=b)
        avg_ssim_bpp_ae = calculate_ssim_bpp(xtest[:NumImagesToShow], decoded_imgsQ, bpp)
        results['AE1_old'].append((b, avg_ssim_bpp_ae))
        


    plt.figure(figsize=(10, 6))
        

    colors = {
        'AE2': 'blue',
        'AE1': 'green',
        'JPEG': 'orange'
    }

    for label, data in results.items():
        b_values, ssim_bpp_values = zip(*data)

        base_color = colors[label.split('_')[0]] if label != 'JPEG' else colors['JPEG']
        
        if '_old' in label:
            color = to_rgba(base_color, alpha=0.7)  # Темнее и более прозрачный оттенок
            linestyle = 'dashed'
            
        elif '_vector' in label:
            color = to_rgba(base_color, alpha=0.8)
            linestyle = 'dotted'
        else:
            color = base_color
            linestyle = '-' 

        plt.plot(
            b_values, ssim_bpp_values, label=label, marker='o', color=color, linestyle=linestyle
        )

    plt.title('Средний SSIM/bpp для различных кодеков', fontsize=14)
    plt.xlabel('b', fontsize=12)
    plt.ylabel('Средний SSIM/bpp', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.show()

def display_compressed_images(enc, dec, bees=[2, 3, 4, 5]):
    """
    Отображает изображения для методов NeuralCompressor_vector, NeuralCompressor_min_max,
    NeuralCompressor и JPEG для заданных значений b.
    """
    fig, axes = plt.subplots(len(bees), 3, figsize=(16, 4 * len(bees)))

    for i, b in enumerate(bees):
        # Вызываем ваши функции для каждого метода
        bpp_vec,  img_vec, Q_vec = NeuralCompressor_min_max(enc, dec, b)
        bpp_minmax, img_minmax, Q_minmax  = NeuralCompressor_vector(enc, dec, b)
        bpp_nc, img_nc, Q_nc = NeuralCompressor(enc, dec, xtest, b=b)
        
        # Отображаем изображения
        images = [Q_vec[0], Q_minmax[0], Q_nc[0]]
        titles = [
            f"Vector: Q={calculate_ssim(Q_vec)}, bpp={np.mean(bpp_vec)}",
            f"MinMax: Q={calculate_ssim(Q_minmax)}, bpp={np.mean(bpp_minmax)}",
            f"NC: Q={calculate_ssim(Q_nc)}, bpp={np.mean(bpp_nc)}",
        ]
        
        for j, (img, title) in enumerate(zip(images, titles)):
            ax = axes[i, j] if len(bees) > 1 else axes[j]  
            ax.imshow(img, cmap='gray')
            ax.set_title(title, fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    show_img = True
    xtest = LoadImagesFromFolder(testfolder)
    xtest = xtest / 255
    
    encoder, decoder = ImageCodecModel(trainfolder,0)
    encoder2, decoder2 = ImageCodecModel(trainfolder,1)
    

    
    bees = [2, 3, 4, 5]
    
    if show_img:
        NumImagesToShow = 1
        
        display_compressed_images(encoder, decoder)  
        display_compressed_images(encoder2, decoder2)  
        
    NumImagesToShow = 10
    plot_ssim_bpp(testfolder, trainfolder, bees)
    
    

