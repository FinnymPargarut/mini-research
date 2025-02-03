from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


def init_model_and_processor():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto",
    )

    min_pixels = 128 * 28 * 28
    max_pixels = 640 * 28 * 28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    return model, processor


def init_inputs(processor, img_path, text):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": text},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu") # maybe model.device
    return inputs


def predict_tokens(model, processor, inputs, tokens_count=10, logging=True):
    input_ids = inputs.input_ids
    pixel_values = inputs.pixel_values
    image_grid_thw = inputs.image_grid_thw
    attention_mask = inputs.attention_mask
    predicted_tokens = []

    # Генерация по одному токену
    for _ in range(tokens_count):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                            attention_mask=attention_mask)
        if logging:
            print(outputs.logits.shape)
        # Получите логиты для последнего токена
        logits = outputs.logits[:, -1, :]
        predicted_token_id = torch.argmax(logits, dim=-1).item()

        # Добавьте предсказанный токен к входной последовательности
        input_ids = torch.cat([input_ids, torch.tensor([[predicted_token_id]], device=input_ids.device)], dim=-1)

        # Преобразуйте predicted_token_id в текст
        predicted_token = processor.decode(predicted_token_id)
        if predicted_token == '<|im_end|>':
            break
        predicted_tokens.append(predicted_token)
    if logging:
        print("Predicted tokens:", predicted_tokens)
    return predicted_tokens
