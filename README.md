# CrazyFrogger

## Visão geral

Este projeto faz parte do curso "Projetos em Ciência de Dados", um curso de graduação do 7º semestre do programa de Ciência de Dados. Nosso objetivo é desenvolver um mecanismo de pesquisa adaptado a dados de música, com foco em letras, artistas e gêneros. Utilizando o abrangente banco de dados de música, nossos esforços estarão concentrados em aprimorar a experiência do usuário ao melhorar a eficiência da pesquisa utilizando tecnologias de modelos de linguagem.


## Metas do projeto

- Criar um mecanismo de busca eficiente e fácil de usar para consultas relacionadas a música.

- Aproveitar o banco de dados MusicBrainz para fornecer informações detalhadas sobre letras, artistas e gêneros.

- Implementar algoritmos de pesquisa avançados que possam corresponder às preferências musicais dos usuários.

- Explorar a possibilidade de integrar ao Spotify para uma experiência mais imersiva.

## Instruções de uso

O projeto conta com alguns scripts, um servidor e uma interface web.

- Gerar índice para um certo modelo a partir do conjunto de dados:

```sh
$ python scripts/generate-index.py            # Treinar no modelo definido como default
$ python scripts/generate-index.py -m <model> # Treinar para um modelo em específico
$ python scripts/generate-index.py -h         # Mostra a ajuda
```

- Fazer uma busca:

```sh
$ python scripts/run-query.py            # Fazer uma busca no modelo definido como defaul
$ python scripts/run-query.py -m <model> # Fazer uma busca em um modelo em específico
$ python scripts/run-query.py -l         # Busca em modo loop (para fazer múltiplas buscas)
$ python scripts/run-query.py -h         # Mostra a ajuda
```

- Executar os testes para um modelo:

```sh
$ python scripts/test-model.py            # Test o modelo definido como default
$ python scripts/test-model.py -m <model> # Test um modelo em específico
$ python scripts/test-model.py -f 0.5     # Testa em apenas metade do dataset (50%)
$ python scripts/test-model.py -h         # Mostra a ajuda
```

- Executar todos os testes:

```sh
$ ./scripts/run-tests.sh
```

- Iniciar o servidor:

```sh
$ cd scripts && uvicorn server:app # Documentação em http://127.0.0.1:8000/docs
```

- Inciar a interface web:
```sh
$ cd web && npm i && npm run dev # Acesso em http://127.0.0.1:5173
```

## Time
- [Amanda de Mendonça Perez](https://github.com/Perez-Amanda)
- [Breno Marques Azevedo](https://github.com/Breno-Azevedo)
- [Eduardo Adame Salles](https://adamesalles.github.io)
- [Juan Belieni](https://github.com/juanbelieni)
- [Kayo Yokoyama Reis](https://github.com/kayo-ko)
