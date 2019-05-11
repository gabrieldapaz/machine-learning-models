import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM


#carrega um arquivo csv que contem 100k avaliações de 1k users em 1700 filmes
#cada usuario avaliou pelo menos 20 filmes de 1 a 5
data = fetch_movielens(min_rating=4.0)
#esse metodo vai criar uma matriz de interção e guardar
#em data como um dicionario
#o metodo tambem quebra o arquivo dataset de treino e de teste

print(repr(data['train']))
print(repr(data['test']))

#o parametro loss mede a difrença entre o output e o que o modelo previu

#criando o modelo
model = LightFM(loss ='warp')

#o modelo fica mais acurado a cada treino, existem varias opcoes,mas a usada
#é o warp (weighted approximate-rank pairwise), ajuda a criar recomendações para cada usario olhando para o usuario
#pares de avaliacoes e prevendo rankins para cada
#usa o gradient descet algorithm para a cada iteracao que melhorar o pesos das prediçoes com o tempo
#eh um sistema que analisa tanto o contet _ collaborative = hybrid

#treinando o modelo
model.fit(data['train'], epochs=30, num_threads=2)


def sample_recommendation(model, data, user_ids):

	n_users, n_items = data['train'].shape

	for user_id in user_ids:

		know_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

		scores = model.predict(user_id, np.arange(n_items))

		top_itens = data['item_labels'][np.argsort(-scores)]

		print("User %s" % user_id)
		print("      Know positives:")

		for x in know_positives[:3]:
			print("            %s" %x)

		print("       Recommended")

		for x in top_itens[:3]:
			print("          %s" % x)

sample_recommendation(model, data, [3,25,450])