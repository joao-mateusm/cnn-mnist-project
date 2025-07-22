# Kagle: Digit Recognizer com CNN

Este projeto implementa uma Rede Neural Convolucional (CNN) para classificar dígitos manuscritos do dataset MNIST, como parte da competição do Kaggle "Digit Recognizer".

## Estrutura do Projeto
- **/data**: Contém os dados brutos da competição.
- **/notebooks**: Notebooks Jupyter para análise exploratória e apresentação do fluxo de trabalho.
- **/src**: Scripts Python com código modularizado (pré-processamento, modelo, avaliação).
- **/submissions**: Arquivo de submissão gerado.

## Como Executar
1. Clone este repositório: `git clone https://github.com/seu-usuario/digit-recognizer.git`
2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Instale as dependências: `pip install -r requirements.txt`
4. Execute o notebook em `notebooks/digit_recognizer_eda_e_modelo.ipynb`.

## Resultados
O modelo alcançou uma acurácia de **99.67%** no conjunto de validação após 30 épocas de treinamento, utilizando Data Augmentation.