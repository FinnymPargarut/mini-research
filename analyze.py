from model import init_model_and_processor, init_inputs
from loggit_utils import *

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


model, processor = init_model_and_processor()


def get_target_feature_and_idx():
    target_features = {
        "colors": [
            "red", "blue", "green", "yellow", "purple", "cyan",
            "gray", "grey", "brown", "orange", "pink", "magenta",
            "black", "white", "dark blue", "light blue", "navy",
            "teal", "dark green", "light green"
        ],
        "shapes": [
            "cube", "sphere", "ball", "cylinder", "cone", "torus",
            "pyramid", "rectangle", "oval", "dome", "hemisphere",
            "prism", "ring", "disc", "square", "circle", "triangle",
            "hexagon", "octagon", "star", "diamond"
        ]
    }
    feature_token_ids = {
        category: {
            feature: processor.tokenizer.convert_tokens_to_ids(
                processor.tokenizer.tokenize(feature)[0]
            )
            for feature in features
        }
        for category, features in target_features.items()
    }
    return target_features, feature_token_ids


def process_example(image_path, question, max_tokens=10, logging=True):
    inputs = init_inputs(processor, img_path=image_path, text=question)

    # Получение логитов на всех слоях
    intermediate_logits = get_intermediate_logits_dict(model, processor, inputs, max_tokens, logging=logging)

    # Извлечение данных
    layer_logits = get_layer_logits(intermediate_logits, logging=logging)
    layer_tokens = get_layer_tokens(intermediate_logits, processor)

    return {
        "layer_logits": layer_logits,
        "layer_tokens": layer_tokens
    }

def get_general_questions():
    general_questions = [
        "What is the theory of relativity?",
        "How does GPS work?",
        "What is DNA?",
        "Which planet is closest to the Sun?",
        "What is quantum entanglement?",
        "How are black holes formed?",
        "What is photosynthesis?",
        "What is the boiling point of water?",
        "What is artificial intelligence?",
        "How does blockchain work?",
        "Which river is the longest in the world?",
        "What is the capital of Canada?",
        "Where is Everest located?",
        "Which country produces the most coffee?",
        "Which ocean is the largest?",
        "Which desert is the hottest?",
        "How many continents are there on Earth?",
        "What is the capital of Australia?",
        "Where is Machu Picchu located?",
        "Which country is shaped like a boot?",
        "Who discovered America?",
        "When did World War II begin?",
        "Who was the first person on the Moon?",
        "What is the Renaissance?",
        "Who wrote the 'Declaration of Independence'?",
        "When was the Great Wall of China built?",
        "Who was Joan of Arc?",
        "When did the USSR collapse?",
        "Who invented the printing press?",
        "Who wrote 'War and Peace'?",
        "What is a sonnet?",
        "Who painted the 'Mona Lisa'?",
        "Which book is the best-selling in history?",
        "Who is the author of 'Harry Potter'?",
        "What is Impressionism?",
        "Who wrote '1984'?",
        "What is the name of the main character in 'Crime and Punishment'?",
        "What is a haiku?",
        "Who is Shakespeare?",
        "What is freedom?",
        "What is the meaning of life?",
        "What is morality?",
        "How do you define consciousness?",
        "What is the liar's paradox?",
        "What is dialectics?",
        "What is déjà vu?",
        "What is happiness?",
        "How does memory work?",
        "What is time?",
        "How do you cook pasta carbonara?",
        "What is sushi?",
        "How do you make the perfect omelette?",
        "What is fermentation?",
        "Which country invented pizza?",
        "What is guacamole?",
        "How do you brew coffee in a cezve?",
        "What is molecular gastronomy?",
        "How do you bake a chocolate cake?",
        "What is pesto?",
        "Which country won the FIFA World Cup in 2022?",
        "Who is LeBron James?",
        "What is Wimbledon?",
        "How do you play poker?",
        "What is a marathon?",
        "Who is the most decorated Olympian?",
        "What are the rules of basketball?",
        "What is Formula 1?",
        "Who is Lionel Messi?",
        "How do you play chess?",
        "What is an API?",
        "How does a neural network work?",
        "What is cryptocurrency?",
        "How do you create a website?",
        "What is Big Data?",
        "What is the metaverse?",
        "How does facial recognition work?",
        "What is an algorithm?",
        "What is cloud computing?",
        "How do you protect data on the internet?",
        "How many planets are in the Solar System?",
        "Who invented the telephone?",
        "What is gravity?",
        "What is the largest country in the world?",
        "Who wrote 'Das Kapital'?",
        "What is an MRI?",
        "How does a microwave work?",
        "Who is Albert Einstein?",
        "What is a photon?",
        "How is a rainbow formed?",
        "What came first: the chicken or the egg?",
        "What does silence sound like?",
        "What is infinity?",
        "Is time travel possible?",
        "What is love?",
        "Why is the sky blue?",
        "How does intuition work?",
        "What is déjà vu?",
        "What is dark matter?",
        "What is quantum superposition?"
    ]
    return general_questions


def analyze_dataset(dataset, feature_token_ids, num_examples=1000):
    num_layers = model.config.num_hidden_layers
    max_tokens = 10

    all_logits_visual = {
        "colors": np.zeros((num_layers, max_tokens)),
        "shapes": np.zeros((num_layers, max_tokens))
    }
    all_logits_general = {
        "colors": np.zeros((num_layers, max_tokens)),
        "shapes": np.zeros((num_layers, max_tokens))
    }
    counts_visual = np.zeros((num_layers, max_tokens))
    counts_general = np.zeros((num_layers, max_tokens))

    for example in tqdm(dataset[:num_examples]):
        result_visual = process_example(example["image_path"], example["visual_question"])
        result_general = process_example(example["image_path"], example["general_question"])

        # Обработка визуального вопроса
        for layer_idx in range(result_visual["layer_logits"].shape[0]):
            for token_idx in range(result_visual["layer_logits"].shape[1]):
                token_str = result_visual["layer_tokens"][layer_idx][token_idx].strip()
                # Токенизируем с пробелом для соответствия feature_token_ids
                processed_tokens = processor.tokenizer.tokenize(token_str)
                if not processed_tokens:
                    continue
                processed_token_id = processor.tokenizer.convert_tokens_to_ids(processed_tokens[0])

                for category in ["colors", "shapes"]:
                    # Проверяем, есть ли ID в текущей категории
                    if processed_token_id in feature_token_ids[category].values():
                        # feature_name = [k for k, v in feature_token_ids[category].items() if v == processed_token_id][0]
                        # print("BBBBBBBBBB", feature_name)
                        # Суммируем логиты и увеличиваем счётчик
                        all_logits_visual[category][layer_idx, token_idx] += result_visual["layer_logits"][
                            layer_idx, token_idx]
                        counts_visual[layer_idx, token_idx] += 1

        # Аналогично для общего вопроса
        for layer_idx in range(result_general["layer_logits"].shape[0]):
            for token_idx in range(result_general["layer_logits"].shape[1]):
                token_str = result_general["layer_tokens"][layer_idx][token_idx]
                processed_tokens = processor.tokenizer.tokenize(" " + token_str)
                if not processed_tokens:
                    continue
                processed_token_id = processor.tokenizer.convert_tokens_to_ids(processed_tokens[0])

                for category in ["colors", "shapes"]:
                    if processed_token_id in feature_token_ids[category].values():
                        all_logits_general[category][layer_idx, token_idx] += result_general["layer_logits"][
                            layer_idx, token_idx]
                        counts_general[layer_idx, token_idx] += 1

    return all_logits_visual, all_logits_general


def visualize_logits(logits_data_visual, logits_data_general, category):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Визуализация для визуальных вопросов
    sns.heatmap(
        logits_data_visual[category].T,
        cmap="viridis",
        xticklabels=10,
        yticklabels=5,
        cbar_kws={'label': 'Average Logit Value'},
        ax=axes[0]
    )
    axes[0].invert_yaxis()  # Инвертировать ось Y
    axes[0].set_title(f"Visual Questions: Layer-wise Logit Activation for {category.capitalize()}")
    axes[0].set_xlabel("Layer Index")
    axes[0].set_ylabel("Token Position")

    # Визуализация для общих вопросов
    sns.heatmap(
        logits_data_general[category].T,
        cmap="viridis",
        xticklabels=10,
        yticklabels=5,
        cbar_kws={'label': 'Average Logit Value'},
        ax=axes[1]
    )
    axes[1].set_title(f"General Questions: Layer-wise Logit Activation for {category.capitalize()}")
    axes[1].set_xlabel("Layer Index")
    axes[1].set_ylabel("Token Position")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    dataset_path = "/home/bulat/.cache/kagglehub/datasets/timoboz/clevr-dataset/versions/2"
    # Загрузка аннотаций
    with open(os.path.join(dataset_path, "CLEVR_v1.0", "scenes", "CLEVR_train_scenes.json"), 'r') as f:
        scenes = json.load(f)['scenes']

    # Загрузка изображений
    image_dir = os.path.join(dataset_path, "images", "train")
    image_files = sorted(os.listdir(image_dir))

    general_questions = get_general_questions()
    target_features, feature_token_ids = get_target_feature_and_idx()


    def prepare_dataset(scenes, image_files, image_dir, num_examples=1000):
        dataset = []

        for i in range(min(num_examples, len(scenes))):
            scene = scenes[i]
            image_path = os.path.join(image_dir, image_files[i])

            # Создаем два типа запросов:
            # 1. Визуальный запрос (анализ изображения)
            visual_question = "What are the shapes and colors of the objects in the image?"

            # 2. Общий запрос (без анализа изображения)
            general_question = np.random.choice(general_questions)

            dataset.append({
                "image_path": image_path,
                "visual_question": visual_question,
                "general_question": general_question
            })

        return dataset


    # Подготовка данных
    dataset = prepare_dataset(scenes, image_files, image_dir, num_examples=100)
    logits_data_visual3, logits_data_general3 = analyze_dataset(dataset, num_examples=100)

    for category in ["colors", "shapes"]:
        logits_data_visual3[category] = logits_data_visual3[category][::-1]
        logits_data_general3[category] = logits_data_general3[category][::-1]

    visualize_logits(logits_data_visual3, logits_data_general3, "colors")
    visualize_logits(logits_data_visual3, logits_data_general3, "shapes")
