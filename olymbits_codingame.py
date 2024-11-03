import sys
import math
import numpy as np
import copy
import random
import time
from datetime import timedelta

class Data:

    def __init__(self, gpu, reg0, reg1, reg2, reg3, reg4, reg5, reg6):
        self.gpu = gpu
        self.reg0 = reg0
        self.reg1 = reg1
        self.reg2 = reg2
        self.reg3 = reg3
        self.reg4 = reg4
        self.reg5 = reg5
        self.reg6 = reg6

    def clone(self):
        dt = Data(self.gpu, self.reg0, self.reg1,self.reg2,self.reg3,self.reg4,self.reg5,self.reg6)
        return dt

gamma=[1.0]
beta=[0.6711908578872681]
netw=[[[0.122298,0.505272,1.440149,-0.081900,0.726651,1.714429,0.562952,-0.186442,-0.805634,1.742566,0.909455,1.397004,-0.359847,0.359373,1.464817,0.970413,-0.772249,0.585589,1.290436,1.152657,-0.678822,0.694394,0.279309,0.364485,-0.020020,0.921385,0.168939,-0.698897,-0.453575,0.415344,0.841742,0.212251]],
[[-0.040330,-0.096156,0.081141,-0.167655],[-0.160324,-0.322390,-0.190397,0.469600],[-0.255129,-0.273750,-0.227896,0.834889],[0.076436,-0.024064,-0.170662,-0.144355],[0.076207,0.072767,-0.122082,-0.058186],[-0.225950,-0.401540,-0.314004,0.717922],[-0.094393,0.047343,-0.070111,-0.125702],[0.162932,-0.157517,0.017976,0.015148],[-0.095185,0.092778,-0.113953,0.033509],[-0.267905,-0.435316,-0.292385,0.703450],[-0.419261,-0.456130,-0.286388,0.617731],[-0.271901,-0.200365,-0.373489,0.831474],[-0.245096,-0.134568,-0.266588,0.346693],[-0.176261,-0.109592,0.028028,0.097780],[-0.455326,-0.389231,-0.354685,0.580947],[-0.083177,-0.209132,-0.094453,0.391417],[0.110204,-0.161282,-0.080172,-0.040019],[0.001412,-0.423857,-0.273288,0.478396],[-0.050812,-0.496765,-0.161033,0.882052],[-0.247567,-0.481779,-0.381407,0.717980],[0.075597,-0.137108,-0.007627,-0.164496],[-0.314114,-0.349428,-0.173261,0.411356],[-0.242494,-0.397620,-0.055478,0.625438],[0.009572,-0.137790,-0.062679,-0.066210],[0.031028,0.040144,-0.157269,-0.126358],[-0.144296,-0.229919,-0.282409,0.688983],[0.096732,0.098660,-0.036465,-0.061667],[-0.030271,-0.028818,-0.040142,-0.013639],[-0.130644,0.020450,-0.071119,-0.048281],[-0.144330,-0.412671,-0.268141,0.805117],[-0.275911,-0.303070,-0.139719,0.795425],[0.046053,-0.158700,-0.051803,-0.017621]]]
netb=[[-0.086238,1.015710,0.500486,-0.518134,-0.591107,1.509922,-0.519052,-0.035970,-0.249627,1.167653,0.569938,0.676253,1.242395,-0.667736,0.276446,0.939232,-0.268541,0.817950,0.489078,0.670827,-0.052662,0.841670,0.749250,-0.283339,-0.264480,0.949932,-0.711345,-0.978106,-0.233303,0.609379,0.652606,-0.590290],
[-0.065867,-0.220258,-0.314984,0.372353]]



class NeuralNetwork:

	def __init__(self, dimension, LR):
		self.network  =[]
		self.network_w = netw
		self.network_b = netb
		self.gamma = gamma
		self.beta = beta
		for dim in dimension:
			self.network.append([0.0]*dim)
        

	def Relu(self, x):
		return np.maximum(0, x)

	
	def SetInput(self, inp):
		for i in range(len(self.network[0])):
			self.network[0][i] = inp[i]

	
	def batch_norm_list(self, epsilon=1e-5):
		"""
		Applique la normalisation de lots sur un réseau sous forme de liste.
	
		Arguments:
		network_layer -- Une liste de valeurs représentant une couche du réseau (ex: self.network[0])
		gamma -- Facteur d'échelle (1D array ou liste)
		beta -- Décalage (1D array ou liste)
		epsilon -- Constante pour éviter la division par zéro (stabilité numérique)
	
		Retour:
		normalized_layer -- Liste normalisée et transformée.
		"""
		# Convertir la liste en tableau NumPy
		x = np.array(self.network[0])
	
		# Calcul de la moyenne et de la variance pour chaque caractéristique
		batch_mean = np.mean(x, axis=0)
		batch_var = np.var(x, axis=0)
	
		# Normaliser les valeurs
		x_normalized = (x - batch_mean) / np.sqrt(batch_var + epsilon)
	
		# Appliquer l'échelle gamma et le décalage beta
		y = self.gamma * x_normalized + self.beta
	
		# Convertir de nouveau en liste
		self.network[0] = y.tolist()
		
	

	def PredictNN(self):

		for i in range(0, len(self.network)-2):
			for j in  range(0, len(self.network[i+1])):
				h = 0.0
				for k in range(0, len(self.network[i])):
					h += self.network[i][k] * self.network_w[i][k][j]

				h += self.network_b[i][j];
				self.network[i+1][j] = self.Relu(h)#self.LRelu(h, 0.01);


		ind = len(self.network)-2
		for j in range(len(self.network[ind+1])):
			h = 0.0
			for k in range(len(self.network[ind])):
				h += self.network[ind][k] * self.network_w[ind][k][j]
			h += self.network_b[ind][j];
			self.network[ind + 1][j] = h#math.tanh(h)#self.LRelu(h, 0.01);

		ind = len(self.network)-1
		indt= -1;
		maxi = -2000000000
		for j in range(4):
			#print(j, ' ', self.network[ind][j])
			if self.network[ind][j] > maxi :
				maxi = self.network[ind][j]
				
				indt = j;

		return indt;


def Prepare_data(data : Data):

    pos = data.reg0
    map = data.gpu
    st = data.reg3

    state = []
    #state = [ord(char)/50.0 for char in map]
    #state.append(pos / 30.0)

    posh = 30
    i = pos
    while i < 30:
        if map[i] == '#':
            posh = i
            break
        i += 1

    state.append((posh-pos) / 10.0)

    print("state=", len(state), file=sys.stderr, flush=True)

    return state


class Hurdle:

	PLAYER_COUNT = 4  # Define the player count as needed

	def __init__(self):
		self.stun_timers = 0
		self.positions = 0
		self.finished = -1
		self.rank = 0
		self.jumped = False
		self.map = ""

	def clone(self):
		return copy.deepcopy(self)

	def reset(self):
		# Generate new map
		start_stretch = 3 + random.randint(0, 4)
		hurdles = 3 + random.randint(0, 3)
		length = 30

		map_builder = []
		map_builder.extend('.' * start_stretch)
		for _ in range(hurdles):
			if random.choice([True, False]):
				map_builder.append("#....")
			else:
				map_builder.append("#...")

		# Ensure map has exactly the specified length
		while len("".join(map_builder)) < length:
			map_builder.append(".")

		self.map = "".join(map_builder)[:length - 1] + "."

		# Reset player attributes
		
		self.stun_timers = 0
		self.positions = 0
		self.finished = -1
		self.jumped = False
		self.rank = 0

	def step(self, done):

		if self.positions >= 30:
			done[0] = 1
			
		return 1.0

	def choose_action(self):
		length = len(self.map)

		# Define possible actions and the corresponding positions they would move to
		actions = {
			'UP': self.positions + 2,
			'RIGHT': self.positions + 3,
			'DOWN': self.positions + 2,
			'LEFT': self.positions + 1
		}

		# Check each action in priority order and determine if it's a viable option
		if self.map[self.positions + 1:self.positions + 4].count('#') == 0:
			self.positions = actions['RIGHT']
			return 0  # Prefer moving 3 steps if there are no hurdles

		if self.map[actions['UP']] != '#':
			self.positions = actions['UP']
			return 1  # Jumping over hurdles is preferred if possible
		
		if self.map[self.positions + 1:self.positions + 3].count('#') == 0:
			self.positions = actions['DOWN']
			return 2  # Move 2 steps if clear

		if self.map[actions['LEFT']] != '#':
			self.positions = actions['LEFT']
			return 3  # Move 1 step as a last resort

		return 10


class Arc:
	def __init__(self, x, y, wind):
		# Initialisation des attributs de la classe
		self.player = [x, y]
		self.wind = wind

	def apply_wind_effect(self, x, y, vx, vy, direction):
		"""Applique la direction choisie et le vent pour calculer la nouvelle position."""
		if direction == 0:  # right
			return x + vx, y
		elif direction == 3:  # left
			return x - vx, y
		elif direction == 1:  # up
			return x, y - vy
		elif direction == 2:  # down
			return x, y + vy

		return 0, 0

	def euclidean_distance(self, x, y):
		"""Calcule la distance euclidienne entre (x, y) et (0, 0)."""
		return math.sqrt(x ** 2 + y ** 2)

	def Get_dir(self):
		w = int(self.wind[0])
		distances = []
		
		#print("x y=", self.player[0],' ' ,self.player[1], file=sys.stderr, flush=True)
		# Calculer les distances pour chaque direction
		for i in range(4):
			x, y = self.apply_wind_effect(self.player[0], self.player[1], w, w, i)
			#print("x y=", x, ' ', y, file=sys.stderr, flush=True)

			distance = int(self.euclidean_distance(x, y))
			distances.append((i, distance))  # Stocker la direction et sa distance
			#print("mind=", distance, file=sys.stderr, flush=True)


		# Trouver la distance minimale
		min_distance = min(dist for _, dist in distances)
		#print("mind=", min_distance, file=sys.stderr, flush=True)

		# Récupérer toutes les directions ayant cette distance minimale
		best_directions = [direction for direction, dist in distances if dist == min_distance]

		return best_directions


class Dive:

	def __init__(self, gpu):
		self.gpu = gpu

	def get_action(self, ind):
		actions = {
			'U': 1,
			'R': 0,
			'D': 2,
			'L': 3
		}
		return actions[self.gpu[ind]]



MAX_DOUBLE = sys.float_info.max

class TNode:
	def __init__(self):
		self.child = []  # Liste pour les nuds enfants, initialisee avec None
		self.ind_child = 0        # Indice du nud enfant
		self.parent = None        # Pointeur vers le noeud parent
		self.ucb = 0.0            # Valeur UCB
		self.n = 0.0              # Nombre de visites
		self.w = 0.0              # Recompense cumulative
		self.num = 0              # Compteur
		self.score = 0            # Score
		self.expand = False        # Indicateur d'expansion

def selection(node, leaf):
	
	for i in range(len(node.child)):
		if int(node.child[i].n) != 0:
			ad = math.sqrt(2.0 * math.log(node.n) / node.child[i].n)
			node.child[i].ucb = (node.child[i].score / node.child[i].n) + ad
		else:
			node.child[i].ucb = MAX_DOUBLE  # Utilise la constante pour la plus grande valeur possible
	

	# Selectionner l'enfant avec le UCB le plus eleve
	max_ucb = -MAX_DOUBLE  # Initialise a la plus petite valeur possible
	indi = -1
	for i in range(len(node.child)):
		if node.child[i].ucb > max_ucb:
			max_ucb = node.child[i].ucb
			indi = i
		

	#leaf[0] = node.child[indi]  # On utilise une liste pour permettre la modification par reference
	return indi

def expandn(node, g, d, nbh, nba, nbd):
	node.expand = True

	for i in range(4):
		n = TNode()  
		n.parent = node
		n.num = i
		n.ind_child = 0
		n.score = 0

		node.child.append(n)

def expand(node, g, d, nbh, nba, nbd):
	node.expand = True

	actions = {
		1: g.hurdle.positions + 2,
		0: g.hurdle.positions + 3,
		2: g.hurdle.positions + 2,
		3: g.hurdle.positions + 1
	}

	for i in range(4):
		target_pos = actions[i]
		path_segment = g.hurdle.map[g.hurdle.positions + 1:target_pos + 1]

		
		if g.hurdle.map != 'GAME_OVER' and( (i!= 1 and path_segment.count('#') != 0) or ((actions[i] < len(g.hurdle.map) and i == 1 and g.hurdle.map[actions[i]] == '#')) ):
			p = 0.001
			if nbh >= 3: p = 0.75
			#if random.random() >= p:
			continue

		else:
			if (g.dive.gpu != 'GAME_OVER' and d < len(g.dive.gpu) and g.dive.get_action(d) != i):
				p = 0.1
				if nbd >= 3: p = 0.75
				if random.random() >= p:
					continue

			if d < len(g.arc.wind)  and g.arc.wind != 'GAME_OVER':
				w = int(g.arc.wind[d])
				x , y = g.arc.player[0], g.arc.player[1]

				d1 = g.arc.euclidean_distance(x, y)

				if i == 0:  # right
					x, y =  x + w, y
				elif i == 3:  # left
					x, y =  x - w, y
				elif i == 1:  # up
					x, y =  x, y - w
				elif i == 2:  # down
					x, y =  x, y + w

				d2 = g.arc.euclidean_distance(x, y)

				if d2 > 10.0 and d1 <= 10.0:
					p = 0.1
					if nba >=3:p = 0.75
					#if random.random() >= p:
					continue


		n = TNode()  
		n.parent = node
		n.num = i
		n.ind_child = 0
		n.score = 0

		node.child.append(n)

	if g.dive.gpu != 'GAME_OVER' and d < len(g.dive.gpu) and len(node.child) == 0:
		n = TNode()  
		n.parent = node
		n.num = g.dive.get_action(d)
		n.ind_child = 0
		n.score = 0

		node.child.append(n)
	elif g.hurdle.map != 'GAME_OVER' and len(node.child) == 0:
		n = TNode()  
		n.parent = node
		n.num = g.hurdle.choose_action()
		n.ind_child = 0
		n.score = 0

		node.child.append(n)

	elif len(node.child) == 0:
		n = TNode()  
		n.parent = node
		n.num = random.randint(0, 3)
		n.ind_child = 0
		n.score = 0

		node.child.append(n)

		

def backpropagation(parent, sc: int):
	par = parent
	
	while par is not None:
		
		par.n += 1
		par.score += sc
		par.w += 1
		par = par.parent

def backpropagationd(parent, sc: int, level):
	par = parent
	
	lv = level
	while par is not None:
		
		par.n += 1
		if lv == level:
			par.score += sc
			par.w += 1
		else:
			par.score -= sc

		lv-=1
		if lv == -1:
			lv = 2
		par = par.parent

def apply_wind_effect(x, y, vx, vy, direction):
	"""Applique la direction choisie et le vent pour calculer la nouvelle position."""
	if direction == 0:  # right
		return x + vx, y
	elif direction == 3:  # left
		return x - vx, y
	elif direction == 1:  # up
		return x, y - vy
	elif direction == 2:  # down
		return x, y + vy

	return 0, 0

def choose_game(stun):
    # Génère un nombre aléatoire entre 0 et 1
    rand = random.random()

    if rand < 0.15 and stun == 0:
        return 0   # 15% de chance
    elif rand < 0.55:               # 15% + 40% = 55%
        return 1        # 40% de chance
    else:
        return -1       # 45% de chance

class Game:

	def __init__(self):
		self.arc = Arc(0, 0, '')
		self.hurdle = Hurdle()
		self.dive = Dive('')
		self.scoreh = 0

	def reset(self, data):
		self.arc  = Arc(data[1].reg0, data[1].reg1, data[1].gpu)

		self.hurdle = Hurdle()
		self.hurdle.positions = data[0].reg0
		self.hurdle.map = data[0].gpu
		self.hurdle.stun_timers = data[0].reg3

		self.dive = Dive(data[3].gpu)

		self.data = data
		self.scoreh = 0

	def Advance(self, inds, depth2)-> int:
		score = 0
		if self.hurdle.map != 'GAME_OVER':
			actions = {
				1: self.hurdle.positions + 2,
				0: self.hurdle.positions + 3,
				2: self.hurdle.positions + 2,
				3: self.hurdle.positions + 1
			}
			if self.hurdle.stun_timers > 0:
				self.hurdle.stun_timers -= 1
			elif self.hurdle.positions < 30:
				posh = self.hurdle.map[self.hurdle.positions + 1:actions[inds]+1].find('#')
				if (i != 1 and self.hurdle.map[self.hurdle.positions + 1:actions[inds]+1].count('#') == 0) or (inds == 1 and self.hurdle.map[actions[inds]] != '#'):
					self.hurdle.positions = actions[inds]
					ind = self.hurdle.choose_action()
					if ind == inds:
						score  += self.hurdle.positions	+100
						self.scoreh += 2
					else:
						score  += self.hurdle.positions
						self.scoreh += 1
				else:
					self.hurdle.positions = posh
					self.hurdle.stun_timers = 3
					score  += 0
					self.scoreh -= 10
			else:
				self.hurdle.map = 'GAME_OVER'

		if self.arc.wind != 'GAME_OVER' and depth2 < len(self.arc.wind):
				
			w = int(self.arc.wind[depth2])
			x, y = apply_wind_effect(self.arc.player[0], self.arc.player[1], w, w, inds)
			self.arc.player = [x, y]
			score += 20 - math.sqrt(self.arc.player[0]**2 + self.arc.player[1]**2)
		elif depth2 >= len(self.arc.wind):
			self.arc.wind = 'GAME_OVER'

		if  depth2 < len(self.data[3].gpu) and self.dive.gpu != 'GAME_OVER':
			indsd = self.dive.get_action(depth2)
			if inds == indsd:
				self.data[3].reg3 = self.data[3].reg3 + 1
				#score += 20
			else:
				self.data[3].reg3 = 0
			self.data[3].reg0 = (self.data[3].reg0 + self.data[3].reg3) 
			score+= self.data[3].reg0

		elif  depth2 >= len(self.data[3].gpu):
			self.dive.gpu = 'GAME_OVER'

		return score

	def get_hurdle_score(self):
		return self.scoreh;

	def get_arc_score(self):
		return 20 - math.sqrt(self.arc.player[0]**2 + self.arc.player[1]**2)

	def get_dive_score(self):
		return self.data[3].reg0

class Game2:

	def __init__(self):
		self.arc = Arc(0, 0, '')
		self.hurdle = Hurdle()
		self.dive = Dive('')
		self.scoreh = 0

	def reset(self, data):
		self.arc  = Arc(data[1].reg2, data[1].reg3, data[1].gpu)

		self.hurdle = Hurdle()
		self.hurdle.positions = data[0].reg1
		self.hurdle.map = data[0].gpu
		self.hurdle.stun_timers = data[0].reg4

		self.dive = Dive(data[3].gpu)

		self.data = data
		self.scoreh = 0

	def Advance(self, inds, depth2)-> int:
		score = 0
		if self.hurdle.map != 'GAME_OVER':
			actions = {
				1: self.hurdle.positions + 2,
				0: self.hurdle.positions + 3,
				2: self.hurdle.positions + 2,
				3: self.hurdle.positions + 1
			}
			if self.hurdle.stun_timers > 0:
				self.hurdle.stun_timers -= 1
			elif self.hurdle.positions < 30:
				posh = self.hurdle.map[self.hurdle.positions + 1:actions[inds]+1].find('#')
				if (i != 1 and self.hurdle.map[self.hurdle.positions + 1:actions[inds]+1].count('#') == 0) or (inds == 1 and self.hurdle.map[actions[inds]] != '#'):
					self.hurdle.positions = actions[inds]
					ind = self.hurdle.choose_action()
					if ind == inds:
						score  += self.hurdle.positions	+100
						self.scoreh += 2
					else:
						score  += self.hurdle.positions
						self.scoreh += 1
				else:
					self.hurdle.positions = posh
					self.hurdle.stun_timers = 3
					score  += 0
					self.scoreh -= 10
			else:
				self.hurdle.map = 'GAME_OVER'
			
		if self.arc.wind != 'GAME_OVER' and depth2 < len(self.arc.wind):
				
			w = int(self.arc.wind[depth2])
			x, y = apply_wind_effect(self.arc.player[0], self.arc.player[1], w, w, inds)
			self.arc.player = [x, y]
			score += 20 - math.sqrt(self.arc.player[0]**2 + self.arc.player[1]**2)
		elif depth2 >= len(self.arc.wind):
			self.arc.wind = 'GAME_OVER'

		if  depth2 < len(self.data[3].gpu) and self.dive.gpu != 'GAME_OVER':
			indsd = self.dive.get_action(depth2)
			if inds == indsd:
				self.data[3].reg4 = self.data[3].reg4 + 1
				#score += 20
			else:
				self.data[3].reg4 = 0
			self.data[3].reg1 = (self.data[3].reg1 + self.data[3].reg4 ) 
			score += self.data[3].reg1
		elif  depth2 >= len(self.data[3].gpu):
			self.dive.gpu = 'GAME_OVER'

		return score

	def get_hurdle_score(self):
		return self.scoreh;

	def get_arc_score(self):
		return 20 - math.sqrt(self.arc.player[0]**2 + self.arc.player[1]**2)

	def get_dive_score(self):
		return self.data[3].reg1 

class Game3:

	def __init__(self):
		self.arc = Arc(0, 0, '')
		self.hurdle = Hurdle()
		self.dive = Dive('')
		self.scoreh = 0

	def reset(self, data):
		self.arc  = Arc(data[1].reg4, data[1].reg5, data[1].gpu)

		self.hurdle = Hurdle()
		self.hurdle.positions = data[0].reg2
		self.hurdle.map = data[0].gpu
		self.hurdle.stun_timers = data[0].reg5

		self.dive = Dive(data[3].gpu)

		self.data = data
		self.scoreh = 0

	def Advance(self, inds, depth2)-> int:
		score = 0
		if self.hurdle.map != 'GAME_OVER':
			actions = {
				1: self.hurdle.positions + 2,
				0: self.hurdle.positions + 3,
				2: self.hurdle.positions + 2,
				3: self.hurdle.positions + 1
			}
			if self.hurdle.stun_timers > 0:
				self.hurdle.stun_timers -= 1
			elif self.hurdle.positions < 30:
				posh = self.hurdle.map[self.hurdle.positions + 1:actions[inds]+1].find('#')
				if (i != 1 and self.hurdle.map[self.hurdle.positions + 1:actions[inds]+1].count('#') == 0) or (inds == 1 and self.hurdle.map[actions[inds]] != '#'):
					self.hurdle.positions = actions[inds]
					ind = self.hurdle.choose_action()
					if ind == inds:
						score  += self.hurdle.positions	+100
						self.scoreh += 2
					else:
						score  += self.hurdle.positions
						self.scoreh += 1
				else:
					self.hurdle.positions = posh
					self.hurdle.stun_timers = 3
					score  += 0
					self.scoreh -= 10
			else:
				self.hurdle.map = 'GAME_OVER'
			

		if self.arc.wind != 'GAME_OVER' and depth2 < len(self.arc.wind):
				
			w = int(self.arc.wind[depth2])
			x, y = apply_wind_effect(self.arc.player[0], self.arc.player[1], w, w, inds)
			self.arc.player = [x, y]
			score += 20 - math.sqrt(self.arc.player[0]**2 + self.arc.player[1]**2)
		elif depth2 >= len(self.arc.wind):
			self.arc.wind = 'GAME_OVER'

		if  depth2 < len(self.data[3].gpu) and self.dive.gpu != 'GAME_OVER':
			indsd = self.dive.get_action(depth2)
			if inds == indsd:
				self.data[3].reg5 = self.data[3].reg5 + 1
				#score += 20
			else:
				self.data[3].reg5 = 0
			self.data[3].reg2 = self.data[3].reg2 + self.data[3].reg5
			score += self.data[3].reg2 
		elif  depth2 >= len(self.data[3].gpu):
			self.dive.gpu = 'GAME_OVER'

		return score

	def get_hurdle_score(self):
		return self.scoreh;

	def get_arc_score(self):
		return 20 - math.sqrt(self.arc.player[0]**2 + self.arc.player[1]**2)

	def get_dive_score(self):
		return self.data[3].reg2 

def calculate_mini_game_score(sorted_results):
    medals = {0: (0, 0), 1: (0, 0), 2: (0, 0)}  # {player_index: (gold, silver)}
    
    # Attribuer les médailles (or et argent) en fonction du classement
    if len(sorted_results) > 0:
        medals[sorted_results[0][1]] = (1, 0)  # 1 médaille d'or pour le premier
    if len(sorted_results) > 1:
        medals[sorted_results[1][1]] = (0, 1)  # 1 médaille d'argent pour le deuxième

    # Calculer le score de chaque joueur pour ce mini-jeu
    scores = {}
    for player, (gold, silver) in medals.items():
        scores[player] = silver + gold * 3

    return scores


def Play(depth, data, ind, medals, player_ido, gld)-> int:

	

	root = TNode()
	root.parent = None
	root.score = 0

	root2 = TNode()
	root2.parent = None
	root2.score = 0

	root3 = TNode()
	root3.parent = None
	root3.score = 0
	
	times = 0

	g = Game()
	g2 = Game2()
	g3 = Game3()

	end = time.perf_counter() + 0.048

	nbh = medals[player_idx][1]
	nba = medals[player_idx][4]
	nbd = medals[player_idx][10]

	nbh2 = medals[player_ido[0]][1]
	nba2 = medals[player_ido[0]][4]
	nbd2 = medals[player_ido[0]][10]

	nbh3 = medals[player_ido[1]][1]
	nba3 = medals[player_ido[1]][4]
	nbd3 = medals[player_ido[1]][10]

	id = 0

	while time.perf_counter() < end:
		_node_p1 = root
		node_p1 = None
		_node_p2 = root2
		node_p2 = None
		_node_p3 = root3
		node_p3 = None

		dt1 = []
		for d in data:
			dt1.append(d.clone())
		g.reset(dt1)
		dt2 = []
		for d in data:
			dt2.append(d.clone())
		g2.reset(dt2)
		dt3 = []
		for d in data:
			dt3.append(d.clone())
		g3.reset(dt3)

		inds = inds2 = inds3 = 0
		scoref = [0,0,0]
		s1 = 0
		s2 = 0
		s3 = 0
		level = 0
		depth2 = 0
		while time.perf_counter() < end and depth2 < depth:
			
			if len(_node_p1.child) == 0:
				expandn(_node_p1, g, depth2, nbh, nba, nbd)
				game = choose_game(g.hurdle.stun_timers)
	
				
				if game == 0 and g.hurdle.stun_timers == 0 and g.hurdle.map != 'GAME_OVER':
					
					inds = g.hurdle.choose_action()
				
								
				elif game == 1 and depth2 < len(data[3].gpu) and g.dive.gpu != 'GAME_OVER':
					inds = g.dive.get_action(depth2)
				
				else:
					inds = random.randint(0, 3)
				node_p1 = _node_p1.child[inds]
				
			else:
				inds = selection(_node_p1, node_p1)
				node_p1 = _node_p1.child[inds]		
				
		
			if len(_node_p2.child) == 0:
				expandn(_node_p2, g2, depth2, nbh2, nba2, nbd2)
				game = choose_game(g2.hurdle.stun_timers)
				
				if game == 0 and g2.hurdle.stun_timers == 0 and  g2.hurdle.map != 'GAME_OVER':
					
					inds2 = g2.hurdle.choose_action()
					
								
				elif game == 1 and depth2 < len(data[3].gpu) and g2.dive.gpu != 'GAME_OVER':
					inds2 = g2.dive.get_action(depth2)
				
				else:
					inds2 = random.randint(0, 3)
				node_p2 = _node_p2.child[inds2]
			
			else:
				inds2 = selection(_node_p2, node_p2)
				node_p2 = _node_p2.child[inds2]		
				

			if len(_node_p3.child) == 0:
				expandn(_node_p3, g3, depth2, nbh3, nba3, nbd3)
				game = choose_game(g3.hurdle.stun_timers)
				
				if game == 0 and g3.hurdle.stun_timers == 0 and g3.hurdle.map != 'GAME_OVER':
					
					inds3 = g3.hurdle.choose_action()
				
								
				elif game == 1 and depth2 < len(data[3].gpu) and g3.dive.gpu != 'GAME_OVER':
					inds3 = g3.dive.get_action(depth2)
					
				else:
					inds3 = random.randint(0, 3)
				node_p3 = _node_p3.child[inds3]
			
			else:
				inds3 = selection(_node_p3, node_p3)
				node_p3 = _node_p3.child[inds3]		
			
			
			ind_depth = 0
			#while 1:
			s1 += g.Advance(inds, ind_depth)
			s2 += g2.Advance(inds2, ind_depth)
			s3 += g3.Advance(inds3, ind_depth)

			ind_depth += 1

			if (g.hurdle.map == 'GAME_OVER' or g2.hurdle.map == 'GAME_OVER' or g3.hurdle.map == 'GAME_OVER') and \
			(g.arc.wind == 'GAME_OVER' or g2.arc.wind  == 'GAME_OVER' or g3.arc.wind  == 'GAME_OVER') and \
			(g.dive.gpu == 'GAME_OVER' or g2.dive.gpu  == 'GAME_OVER' or g3.dive.gpu  == 'GAME_OVER'):
				break
					
				
			#print('end rollout')
				
			_node_p1 = node_p1
			_node_p2 = node_p2
			_node_p3 = node_p3

			depth2 += 1

			
									

			#if g.hurdle.positions >= 30 or g2.hurdle.positions >= 30 or g3.hurdle.positions >= 30 :
			#	break
			
		#score  = hurdle.positions * 20.0 / 30.0
		#- math.sqrt(arc.player[0]**2 + arc.player[1]**2)

		if time.perf_counter() >= end:break

		mh = sorted([(g.hurdle.positions, player_idx), (g2.hurdle.positions, player_ido[0]), (g3.hurdle.positions, player_ido[1])], key=lambda x: x[0], reverse=True)
		ma = sorted([(g.arc.euclidean_distance(g.arc.player[0], g.arc.player[1]), player_idx),
			(g2.arc.euclidean_distance(g2.arc.player[0], g2.arc.player[1]), player_ido[0]),
			(g3.arc.euclidean_distance(g3.arc.player[0], g3.arc.player[1]), player_ido[1])],  key=lambda x: x[0], reverse=True)
		md = sorted([(g.get_dive_score(), player_idx), (g2.get_dive_score(), player_ido[0]), (g3.get_dive_score(), player_ido[1])],  key=lambda x: x[0], reverse=True)
				
		player_scores = {0: 1, 1: 1, 2: 1}

		mini_game_scores = [mh, ma, md]
		for game_results in mini_game_scores:
			mini_game_score = calculate_mini_game_score(game_results)
			for player, score in mini_game_score.items():
				player_scores[player] *= score
				
				
		for i in range(2):
			scoref[i] += player_scores[i]

		

		ch = 1.0
		ca = 1.0
		cd = 1.0

		if nbh >= 3:ch = 2.0
		if nba >= 3:ca = 2.0
		if nbd >= 3:cd = 4.0

		sp1 = g.get_hurdle_score() + g.get_arc_score() + g.get_dive_score()
		sp2 = g2.get_hurdle_score() + g2.get_arc_score() + g2.get_dive_score()
		sp3 = g3.get_hurdle_score() + g3.get_arc_score() + g3.get_dive_score()

	
		#sp1 += player_scores[player_idx]**2
		#sp2 += player_scores[player_ido[0]]**2
		#sp3 += player_scores[player_ido[1]]**2

		s1, s2, s3 = sp1, sp2, sp3
	
		#sp1 += (scoref[player_idx]) 
		#sp2 += (scoref[player_ido[0]])
		#sp3 += (scoref[player_ido[1]])


		if time.perf_counter() >= end:break
		backpropagation(node_p1, sp1)
		if time.perf_counter() >= end:break
		backpropagation(node_p2, sp2)
		if time.perf_counter() >= end:break
		backpropagation(node_p3, sp3)
		if time.perf_counter() >= end:break

		times += 1

	print("times=", times, file=sys.stderr, flush=True)
		
	indc = -1
	maxscore = -MAX_DOUBLE
	for i in range(len(root.child)):
		score = 0

		if root.child[i].n == 0:
			continue

		score = int(root.child[i].n)

		print(i, score, file=sys.stderr, flush=True)

		if score > maxscore:
			maxscore = score
			indc = i

	res = root.child[indc].num
			

	return res


def Prepare_Game(data, medals, player_ido) -> int:

	global nb_turn 
	global turn 
	global ind_game
	global player_idx

	gold = [(medals[player_idx][1], 0),(medals[player_idx][4], 1),(medals[player_idx][7], 2),(medals[player_idx][10], 3)]
	gold_sorted = sorted(gold, key=lambda x: x[0], reverse=False)
	
	m = 10
	gld = [1]*4
	#for g in gold_sorted:
	gld[gold_sorted[0][1]]= 100
	m = m / 2
		
	action_g = 0
	#max(5, 30-data[0].reg0)
	action_g = Play(5, data, 1, medals, player_ido, gld)
	
	return action_g
	


# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.
nb_turn = 3
turn = 0
ind_game = 0

player_idx = int(input())
nb_games = int(input())

nn = NeuralNetwork([1, 32, 4], 0.1)
actions = {
    0:'RIGHT',
    1:'UP',
    2:'DOWN',
    3:'LEFT'
}

medals = []
medalsopp = []
player_ido = []
# game loop
while True:
    start = time.perf_counter()
    data = Data("",0,0,0,0,0,0,0)
    datat = []

    for i in range(3):
        score_info = input()
        if player_idx == i:
            m = score_info.split(' ')
            m = [int(x) for x in m] 
            medals.append(m)
        else:
            m = score_info.split(' ')
            m = [int(x) for x in m] 
            medals.append(m)
            player_ido.append(i)

    for i in range(nb_games):
        inputs = input().split()
        gpu = inputs[0]
        reg_0 = int(inputs[1])
        reg_1 = int(inputs[2])
        reg_2 = int(inputs[3])
        reg_3 = int(inputs[4])
        reg_4 = int(inputs[5])
        reg_5 = int(inputs[6])
        reg_6 = int(inputs[7])

        #if i == 0:
        #    data = Data(gpu, reg_0, reg_1, reg_2, reg_3, reg_4, reg_5, reg_6)
        datat.append(Data(gpu, reg_0, reg_1, reg_2, reg_3, reg_4, reg_5, reg_6))

    #action = 0
    #if data.gpu != 'GAME_OVER':
    #    state = Prepare_data(data)
    #    nn.SetInput(state)
    #    nn.batch_norm_list(1e-5)
    #    action = nn.PredictNN()

    action = Prepare_Game(datat, medals, player_ido)

    end = time.perf_counter()
    endt = (end - start)
    print('time=', timedelta(seconds=endt), file=sys.stderr, flush=True)
    
    print(actions[action])

    



    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)

"""if hurdle.map != 'GAME_OVER':
				if hurdle.stun_timers > 0:
					hurdle.stun_timers -= 1
				elif hurdle.positions < 30:
					posh = hurdle.map[hurdle.positions + 1:actions[inds]+1].find('#')
					if hurdle.map[hurdle.positions + 1:actions[inds]+1].count('#') == 0:
						hurdle.positions = actions[inds]
					else:
						hurdle.positions = posh
						hurdle.stun_timers = 3
			"""

"""
	if min_game == 0 and data[0].gpu != 'GAME_OVER':
		hurdle = Hurdle()

		hurdle.positions = data[0].reg0
		hurdle.map = data[0].gpu
		hurdle.stun_timers = data[0].reg3

		action_g = hurdle.choose_action()

	elif min_game == 1 and data[1].gpu != 'GAME_OVER':
		action_g = Play(len(data[1].gpu), data, 1)
	elif min_game == 3 and data[3].gpu != 'GAME_OVER':
		dive = Dive(data[3].gpu)
		action_g = dive.get_action()
"""
"""
			game = choose_game(g.hurdle.stun_timers)
	
			
			if game == 0 and g.hurdle.stun_timers == 0 and g.hurdle.map != 'GAME_OVER':
				
				inds = g.hurdle.choose_action()
			
							
			elif game == 1 and depth2 < len(data[3].gpu) and g.dive.gpu != 'GAME_OVER':
				inds = g.dive.get_action(depth2)
			
			else:
				inds = random.randint(0, 3)
				

			
			#if depth2 >= 5:
			game = choose_game(g2.hurdle.stun_timers)
			
			if game == 0 and g2.hurdle.stun_timers == 0 and  g2.hurdle.map != 'GAME_OVER':
				
				inds2 = g2.hurdle.choose_action()
				
							
			elif game == 1 and depth2 < len(data[3].gpu) and g2.dive.gpu != 'GAME_OVER':
				inds2 = g2.dive.get_action(depth2)
			
			else:
				inds2 =random.randint(0, 3)
			
			
			#if depth2 >= 5:
			game = choose_game(g3.hurdle.stun_timers)
			
			if game == 0 and g3.hurdle.stun_timers == 0 and g3.hurdle.map != 'GAME_OVER':
				
				inds3 = g3.hurdle.choose_action()
			
							
			elif game == 1 and depth2 < len(data[3].gpu) and g3.dive.gpu != 'GAME_OVER':
				inds3 = g3.dive.get_action(depth2)
				
			else:
				inds3 = random.randint(0, 3)
			"""