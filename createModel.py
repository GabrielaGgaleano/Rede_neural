import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import Xception

# Configurações do caminho e parâmetros
data_dir = 'PontosTuristicosCuritiba'  # Substitua pelo caminho real da sua base de dados
augmented_data_dir = 'PontosTuristicosCuritibaAugmented'  # Diretório para salvar imagens augmentadas
img_height, img_width = 299, 299  # Dimensões das imagens
batch_size = 16
num_classes = len(os.listdir(data_dir))  # Número de classes no seu conjunto de dados
num_augmented_images = 10  # Número de imagens augmentadas a serem geradas por imagem original

# Criação do diretório para salvar as imagens augmentadas
if not os.path.exists(augmented_data_dir):
    os.makedirs(augmented_data_dir)
    for class_name in os.listdir(data_dir):
        os.makedirs(os.path.join(augmented_data_dir, class_name))

# Padronização das imagens
def standardize_images(image):
    return (image - np.mean(image, axis=0)) / np.std(image, axis=0)

# Data augmentation para gerar imagens augmentadas
datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=standardize_images,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Gerando imagens augmentadas e salvando
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    save_class_dir = os.path.join(augmented_data_dir, class_name)
    for file_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, file_name)
        img = load_img(img_path, target_size=(img_height, img_width))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=save_class_dir, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i >= num_augmented_images:
                break  # Gera o número especificado de imagens augmentadas por imagem original

print('Imagens augmentadas geradas e salvas com sucesso.')

# Usando o novo diretório de dados augmentados para treinamento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1,
    preprocessing_function=standardize_images
)

train_generator = train_datagen.flow_from_directory(
    augmented_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    augmented_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Calculando steps_per_epoch e validation_steps
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

# Transfer Learning com Xception
base_model = Xception(
    include_top=False,
    weights="imagenet",
    input_shape=(img_height, img_width, 3),
    pooling='avg'
)
base_model.trainable = True

# Construção da rede neural
model = Sequential([
    base_model,
    Dense(num_classes, activation='softmax')
])

# Compilação do modelo
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.MeanSquaredError(), 'Precision', 'Recall'])

# Early stopping e learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

# Treinamento do modelo
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=1000,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, lr_scheduler]
)

# Salvar o modelo
model.save('modelo_lugares.keras')

# Salvar o mapeamento de classes
class_indices = train_generator.class_indices
index_to_class = {v: k for k, v in class_indices.items()}
np.save('class_mapping.npy', index_to_class)

# Avaliando o modelo
loss, accuracy, mse, precision, recall = model.evaluate(validation_generator)
print(f'Loss: {loss}, Accuracy: {accuracy}, Mean Squared Error: {mse}, Precision: {precision}, Recall: {recall}')
