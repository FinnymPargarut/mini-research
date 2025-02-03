from model import predict_tokens
import torch
import numpy as np


intermediate_hidden_states = {}
hooks = []


def get_intermediate_logits_dict(model, processor, inputs, reset_hooks=True, logging=True):
    # Сброс состояния при необходимости
    if reset_hooks:
        for hook in hooks:
            hook.remove()
        hooks.clear()
        intermediate_hidden_states.clear()

        # Регистрируем новые хуки
        def hook_fn(module, input, output):
            generation_token_idx = len(intermediate_hidden_states) // 28
            layer_idx = len(intermediate_hidden_states) % 28
            intermediate_hidden_states[f"token_{generation_token_idx}, layer_{layer_idx}"] = output[0].detach()

        for layer in model.model.layers:
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)

    # Генерация токенов и вычисление логитов
    outputs = predict_tokens(inputs, processor, logging=logging)
    if logging:
        print(outputs)

    intermediate_logits = {}
    for layer_name, hidden_state in intermediate_hidden_states.items():
        logits = model.lm_head(hidden_state[:, -1, :])
        intermediate_logits[layer_name] = logits

    return intermediate_logits


def get_layer_logits(intermediate_logits, logging=True):
    # Создаем пустую матрицу логитов
    num_layers = max(int(k.split("layer_")[1]) for k in intermediate_logits.keys()) + 2
    num_tokens = max(int(k.split("token_")[1].split(",")[0]) for k in intermediate_logits.keys()) + 1
    layer_logits = np.zeros((num_layers, num_tokens))
    if logging:
        print(num_layers, num_tokens)

    # Заполняем матрицу
    for layer_name, logits in intermediate_logits.items():
        token_num = int(layer_name.split("token_")[1].split(",")[0])
        layer_num = int(layer_name.split("layer_")[1]) + 1
        # print(torch.max(logits, dim=-1).values.item())
        # print(token_num, layer_num)
        layer_logits[layer_num, token_num] = torch.max(logits, dim=-1).values.item()
    layer_logits = layer_logits[::-1]
    return layer_logits


def get_layer_tokens(intermediate_logits, processor):
    # Создаем пустой список токенов
    num_layers = max(int(k.split("layer_")[1]) for k in intermediate_logits.keys()) + 2
    num_tokens = max(int(k.split("token_")[1].split(",")[0]) for k in intermediate_logits.keys()) + 1
    layer_tokens = [[] for _ in range(num_layers)]
    for _ in range(num_tokens):
        layer_tokens[0].append('')

    # Заполняем
    for layer_name, logits in intermediate_logits.items():
        token_num = int(layer_name.split("token_")[1].split(",")[0])
        layer_num = int(layer_name.split("layer_")[1]) + 1
        # print(layer_num)
        # print(processor.decode(torch.argmax(logits, dim=-1), skip_special_tokens=True))
        layer_tokens[layer_num].append(processor.decode(torch.argmax(logits, dim=-1), skip_special_tokens=True))

    layer_tokens = layer_tokens[::-1]
    return layer_tokens
