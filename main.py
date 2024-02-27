# Imports
import torch
import datasets
import transformers
import numpy as np
from datasets import load_dataset, load_metric
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import TrainingArguments
from transformers import Trainer
import warnings
warnings.filterwarnings('ignore')

''' Processamento das Imagens '''

# Função de processamento de imagem
def process_example(example):
    inputs = feature_extractor(example['image'], return_tensors = 'pt')
    inputs['labels'] = example['labels']
    return inputs

# Função para o mapeamento de lotes de imagens e alpicação do ViTFeatureExtractor
def transform(example_batch):
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors = 'pt')
    inputs['labels'] = example_batch['labels']
    return inputs

# Função para combinar amostras
def collate_fn(batch):

    return {"pixel_values": torch.stack([x['pixel_values'] for x in batch]),
            "labels": torch.tensor([x["labels"] for x in batch])}

# Cálculo da métrica
def compute_metrics(prediction):
    return metric.compute(predictions = np.argmax(prediction.predictions, axis = 1),
                          references = prediction.label_ids)


if __name__ == '__main__':

    ''' Experimentando o Vision Transformer '''

    # Imports
    from PIL import Image
    import requests

    # Imagem de teste
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Carrega a imagem
    image = Image.open(requests.get(url, stream=True).raw)

    # Carrega o feature extractor
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    # Carrega o modelo pé-treinado
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')

    # Aplica o feature extractor na imagem de entrada
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Extrai a saída do modelo (previsão)
    outputs = model(**inputs)

    # Extrai os logits das previsões
    logits = outputs.logits

    # Extrai o logit de maior valor (a previsão mais provável)
    class_index = logits.argmax()

    ''' Fine-Tunning do Modelo Vision Transformer '''
    # Agora, vamos adaptar o modelo pré-treinado ao nosso próprio caso de uso.

    ''' Carregando os Dados Para o Nosso Caso e Uso '''

    # Carrega os dados
    dados = load_dataset("beans")

    # Extrai o dicionário de uma imagem
    exemplo = dados['train'][315]

    # Extrai a imagem
    image = exemplo['image']

    # Extrai os labels
    labels = dados['train'].features['labels']

    ''' Carregando e Aplicando o ViT Feature Extractor '''

    # Repositório do ViT pré-treinado
    repo_id = "google/vit-base-patch16-224-in21k"

    # Importa o ViTFeatureExtractor
    feature_extractor = ViTFeatureExtractor.from_pretrained(repo_id)

    ''' Processamento das Imagens '''

    # Prepara os dados
    prepared_data = dados.with_transform(transform)

    ''' Construção do Módulo de Treino do ViT '''

    # Métrica a ser usada
    metric = load_metric("accuracy")

    # Labels
    labels = dados['train'].features['labels'].names

    # Importa o modelo ViTForImageClassification indicandos os novos labels que serão usados
    modelo = ViTForImageClassification.from_pretrained(repo_id,
                                                       num_labels=len(labels),
                                                       id2label={str(i): c for i, c in enumerate(labels)},
                                                       label2id={c: str(i) for i, c in enumerate(labels)})

    # Argumentos de treino
    training_args = TrainingArguments(output_dir="resultados",
                                      evaluation_strategy='steps',
                                      num_train_epochs=4,
                                      learning_rate=2e-4,
                                      remove_unused_columns=False,
                                      load_best_model_at_end=True)

    # Trainer
    trainer = Trainer(model=modelo,
                      args=training_args,
                      data_collator=collate_fn,
                      compute_metrics=compute_metrics,
                      train_dataset=prepared_data['train'],
                      eval_dataset=prepared_data['validation'],
                      tokenizer=feature_extractor)

    ''' Treinamento e Avaliação do Modelo '''

    train_results = trainer.train()

    # Salva o modelo em disco
    trainer.save_model('modelos')

    # Log das métricas
    trainer.log_metrics('train', train_results.metrics)

    # Salva as métricas
    trainer.save_metrics('train', train_results.metrics)

    # Avaliação do modelo
    metrics = trainer.evaluate(prepared_data['validation'])
    trainer.log_metrics('eval', metrics)
    trainer.save_metrics('eval', metrics)
