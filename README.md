# Kaggle: Digit Recognizer com CNN

Este projeto implementa uma Rede Neural Convolucional (CNN) para classificar dígitos manuscritos do dataset MNIST, como parte da competição do Kaggle "Digit Recognizer".

## Estrutura do Projeto
- **/data**: Contém os dados brutos da competição (ignorado pelo `.gitignore`).
- **`main.ipynb`**: **Notebook Jupyter principal que orquestra todo o pipeline: análise, pré-processamento, treinamento e avaliação.**
- **/src**: Scripts Python com o código modularizado (funções de pré-processamento, arquitetura do modelo, etc.).
- **/submissions**: Contém os arquivos de submissão gerados.
- **/models**: (Opcional) Para salvar os modelos treinados (ignorado pelo `.gitignore`).

## Como Executar
1. Clone este repositório: `git clone <URL-DO-SEU-REPOSITORIO>`
2. Navegue até a pasta do projeto: `cd <NOME-DO-REPOSITORIO>`
3. Crie e ative um ambiente virtual:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Instale as dependências: `pip install -r requirements.txt`
5. **Abra e execute o notebook principal `main.ipynb` (usando Jupyter ou a extensão do VS Code).**

## Resultados
O modelo alcançou uma acurácia de aproximadamente **~99.67%** no conjunto de validação após 30 épocas de treinamento, utilizando *Data Augmentation*.