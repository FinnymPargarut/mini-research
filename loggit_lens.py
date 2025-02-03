from dataset import get_scenes
from model import init_inputs, init_model_and_processor
from loggit_utils import *

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings


# Игнорируем все предупреждения, связанные с шрифтами
warnings.filterwarnings("ignore", category=UserWarning)


def plot_custom_logit_lens(layer_logits, layer_tokens=None, input_tokens_str=None, layer_names=None):
    """
    Визуализирует логиты в стиле Logit Lens.

    :param layer_logits: Матрица логитов (слои x токены)
    :param input_tokens_str: Список текстовых представлений токенов
    :param layer_names: Список имен слоев (опционально)
    """
    # Если имена слоев не указаны, создаем их автоматически
    if layer_names is None:
        layer_names = [f"Layer {i}" for i in range(layer_logits.shape[0])]
    layer_names = layer_names[::-1]

    if input_tokens_str is None:
        input_tokens_str = [f"Token {i}" for i in range(layer_logits.shape[1])]

    # Создаем тепловую карту
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(
        layer_logits,
        annot=layer_tokens,
        fmt = '',
        cmap="viridis",  # Цветовая схема
        ax=ax,
    )

    # Настройка осей
    ax.set_xticks(np.arange(len(input_tokens_str)) + 0.5, minor=False)
    ax.set_xticklabels(input_tokens_str, rotation=90)
    ax.set_yticks(np.arange(len(layer_names)) + 0.5, minor=False)
    ax.set_yticklabels(layer_names, rotation=0)

    # Подписи осей
    ax.set_xlabel("Tokens")
    ax.set_ylabel("Layers")
    ax.set_title("Logit Lens")

    plt.tight_layout()
    plt.show()


# plot_custom_logit_lens(
#     layer_logits=layer_logits,
#     input_tokens_str=input_tokens_str,
#     layer_names=[f"Layer {i}" for i in range(num_layers)]
# )

if __name__ == '__main__':
    model, processor = init_model_and_processor()

    dataset_path = "/home/bulat/.cache/kagglehub/datasets/timoboz/clevr-dataset/versions/2"
    image_path = dataset_path + "/CLEVR_v1.0/images/train/CLEVR_train_000900.png"
    obj = get_scenes()[900]['objects'][0]
    visual_question = f"How many {obj['color']} {obj['shape']}s are there?"
    inputs = init_inputs(processor, img_path=image_path, text=visual_question)

    intermediate_logits = get_intermediate_logits_dict(model, processor, inputs)
    layer_logits = get_layer_logits(intermediate_logits)
    layer_tokens = get_layer_tokens(intermediate_logits, processor)
    plot_custom_logit_lens(layer_logits=layer_logits, layer_tokens=layer_tokens)
