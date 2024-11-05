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
		h = Hurdle()
		h.stun_timers = self.stun_timers
		h.positions = self.positions 
		h.finished  = self.finished 
		h.rank = self.rank
		h.jumped  = self.jumped 
		h.map  = self.map 
		return h

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
		self.depth = 0

	def clone(self):
		a = Arc(self.player[0], self.player[1], self.wind)
		a.depth = self.depth
		return a

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
		w = int(self.wind[self.depth])
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
		self.depth = 0

	def clone(self):
		d = Dive(self.gpu)
		d.depth = self.depth
		return d
		

	def get_action(self, ind):
		actions = {
			'U': 1,
			'R': 0,
			'D': 2,
			'L': 3
		}
		return actions[self.gpu[ind]]

class Roller:

    def __init__(self, gpu):
        self.risk = 0
        self.position = 0  # Ajout d'une position pour suivre la piste
        self.stunned = 0  # Ajout d'un état pour gérer l'étourdissement
        self.gpu = gpu  # Ordre de risque (actions)
        self.rem_turn = 15

    def clone(self):
        # Retourne une copie du Roller avec les mêmes attributs
        cloned_gpu = self.gpu  # Clonage de la liste pour éviter les références partagées
        clone = Roller(cloned_gpu)
        clone.risk = self.risk
        clone.position = self.position
        clone.stunned = self.stunned
        return clone

    def reset(self):
        self.risk = 0
        self.position = 0  # Ajout d'une position pour suivre la piste
        self.stunned = 0 
        self.rem_turn = 15

    def move(self, action_index):
        
		
        if self.stunned > 0:
            self.stunned -= 1
            return

        # Effets de déplacement et de risque pour chaque action
        move_effects = [1, 2, 2, 3]
        risk_effects = [-1, 0, 1, 2]

        # Appliquer les effets basés sur l'indice de l'action
        self.position = (self.position + move_effects[action_index]) % 10
        self.risk = max(0, self.risk + risk_effects[action_index])

        # Vérifier si le risque atteint ou dépasse 5
        if self.risk >= 5:
            self.stunned = 2
            self.risk = 0

        self.rem_turn = 15 - (self.position // 10)
        if self.rem_turn == 0:
            self.gpu = 'GAME_OVER'

        

    def __repr__(self):
        return f"Roller(Position={self.position}, Risk={self.risk}, Stunned={self.stunned}, GPU={self.gpu})"


MAX_DOUBLE = sys.float_info.max

class Tstat:
	def __init__(self):
		self.score = 0 
		self.n = 0.0
		self.create = False

class TNode:
	def __init__(self):
		self.child = [Tstat() for _ in range(4)]  # Liste pour les nuds enfants, initialisee avec None
		self.ind_child = 0        # Indice du nud enfant
		self.parent = None        # Pointeur vers le noeud parent
		self.ucb = 0.0            # Valeur UCB
		self.n = 0.0             # Nombre de visites
		self.w = 0.0              # Recompense cumulative
		self.num = 0              # Compteur
		self.score = 0            # Score
		self.expand = False        # Indicateur d'expansion
		#self.child64 = [[[None for _ in range(4)] for _ in range(4)] for _ in range(4)]
		#self.childp = [[Tstat() for j in range(4)] for i in range(3)]
		self.ind_p1 = -1
		self.ind_p2 = -1
		self.ind_p3 = -1
		self.create = True
		

def selection(node, leaf):
	
	for i in range(len(node.child)):
		if int(node.child[i].n) != 0:
			ad = math.sqrt(2.0 * math.log(node.n) / node.child[i].n)
			node.child[i].ucb = (node.child[i].score / node.child[i].n) + ad
		else:
			node.child[i].ucb = MAX_DOUBLE # Utilise la constante pour la plus grande valeur possible
	

	# Selectionner l'enfant avec le UCB le plus eleve
	max_ucb = -MAX_DOUBLE  # Initialise a la plus petite valeur possible
	indi = 10000000
	for i in range(len(node.child)):
		if node.child[i].ucb > max_ucb:
			max_ucb = node.child[i].ucb
			indi = i
		

	#leaf[0] = node.child[indi]  # On utilise une liste pour permettre la modification par reference
	#print(node.child[indi].ucb)
	return indi

def selections(node, leaf, scale):
	
	for i in range(len(node.child)):
		if int(node.child[i].n) != 0:
			ad = math.sqrt(2.0 * math.log(node.n) / node.child[i].n)
			node.child[i].ucb = (node.child[i].score / ((node.child[i].n * scale)+1e-7)) + ad
		else:
			node.child[i].ucb = node.child[i].score  # Utilise la constante pour la plus grande valeur possible
	

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

    if rand < 0.25 and stun == 0:
        return 0   # 15% de chance
    elif rand < 0.5:               # 15% + 40% = 55%
        return 1        # 40% de chance
    else:
        return -1       # 45% de chance

def choose_game2(stun):
    # Génère un nombre aléatoire entre 0 et 1
    rand = random.random()

    if rand < 0.15 and stun == 0:
        return 0   # 15% de chance
    elif rand < 0.55:               # 15% + 40% = 55%
        return 1        # 40% de chance
    else:
        return -1

def random_choice(random_gen, coeffs):
    rand = random.random()
    total = sum(coeffs)
    weights = [v / total for v in coeffs]
    cur = 0
    for i, weight in enumerate(weights):
        cur += weight
        if cur >= rand:
            return i
    return 0

def reset(random_gen):
    x = (5 + random.randint(0, 4)) * (1 if random.choice([True, False]) else -1)
    y = (5 + random.randint(0, 4)) * (1 if random.choice([True, False]) else -1)
    
    cursor = [0, 0] # Assuming `cursors` is a list of lists
    #for cursor in cursors:
    cursor[0] = x
    cursor[1] = y

    wind = []
    rounds = 12 + random.randint(0, 3)

    weights = [0, 2, 2, 2, 0.5, 0.5, 0.25, 0.25, 0.25, 0.2]
    for _ in range(rounds):
        wind.append(random_choice(random_gen, weights))
    
    # Transform wind into a string
    wind_str = ''.join(map(str, wind))

    #print("wind_str=", wind_str, file=sys.stderr, flush=True)
    
    arrows = False
    return cursor, wind_str, arrows



def Create_arc():
	
	"""s = ''
	n = random.randint(10, 15)
	for i in range(n):
		s+= str(random.randint(1, 9))

	a  = Arc(random.randint(-20, 20), random.randint(-20, 20), s)"""

	random_gen = random.random()
	cursors, wind_str, arrows = reset(random_gen)

	a  = Arc(cursors[0], cursors[1], wind_str)

	return a

def Create_Hurdle():

	h = Hurdle()
	h.reset()

	return h

def Create_Dive():
	s = ''
	n = 12 + random.randint(0, 3)
	a=  {0:'U', 1:'R', 2:'D', 3:'L'}
	for i in range(n):
		s += a[random.randint(0,3)]
	
	d = Dive(s)

	return d


def Create_Tab(Hurdle_state, Arc_state, Dive_state):

	Hurdle_state.clear()
	Arc_state.clear()
	Dive_state.clear()

	for i in range(100):
		Hurdle_state.append(Create_Hurdle())
		Arc_state.append(Create_arc())
		Dive_state.append(Create_Dive())



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

		self.roller = Roller(data[2].gpu)

		self.data = data
		self.scoreh = 0

	def Advance(self, inds)-> int:
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
				# Ensure actions[inds] is a valid index
				if posh == -1:
					self.hurdle.positions = actions[inds]
					ind = self.hurdle.choose_action()
					if ind == inds:
						score  += self.hurdle.positions	+100
						self.scoreh += 2
					else:
						score  += self.hurdle.positions
						self.scoreh += 1
				elif (inds != 1 and self.hurdle.positions + 1 < len(self.hurdle.map) and actions[inds] + 1 <= len(self.hurdle.map) and 
					self.hurdle.map[self.hurdle.positions + 1:actions[inds] + 1].count('#') == 0) or \
				(inds == 1 and actions[inds] < len(self.hurdle.map) and self.hurdle.map[actions[inds]] != '#'):

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
					self.scoreh -= 0
						

		if self.hurdle.positions >= 30:
			self.hurdle.map = 'GAME_OVER'

		if self.arc.wind != 'GAME_OVER' and self.arc.depth < len(self.arc.wind):
				
			w = int(self.arc.wind[self.arc.depth])
			x, y = apply_wind_effect(self.arc.player[0], self.arc.player[1], w, w, inds)
			self.arc.player = [x, y]
			score += 20 - math.sqrt(self.arc.player[0]**2 + self.arc.player[1]**2)
			self.arc.depth += 1
		if self.arc.depth >= len(self.arc.wind):
			self.arc.wind =  'GAME_OVER'
			

		if  self.dive.gpu != 'GAME_OVER' and self.dive.depth < len(self.dive.gpu):
			#print(self.dive.gpu, file=sys.stderr, flush=True)
			indsd = self.dive.get_action(self.dive.depth)
			if inds == indsd:
				self.data[3].reg3 = self.data[3].reg3 + 1
				#score += 20
			else:
				self.data[3].reg3 = 0
			self.data[3].reg0 = (self.data[3].reg0 + self.data[3].reg3) 
			score+= self.data[3].reg0
			self.dive.depth += 1

		if  self.dive.depth >= len(self.data[3].gpu):
			self.dive.gpu = 'GAME_OVER'

		if self.roller.gpu != 'GAME_OVER':
			self.roller.move(inds)
	

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

		self.roller = Roller(data[2].gpu)

		self.data = data
		self.scoreh = 0
	
		

	def Advance(self, inds)-> int:
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
				
				# Ensure actions[inds] is a valid index
				if posh == -1:
					self.hurdle.positions = actions[inds]
					ind = self.hurdle.choose_action()
					if ind == inds:
						score  += self.hurdle.positions	+100
						self.scoreh += 2
					else:
						score  += self.hurdle.positions
						self.scoreh += 1

				elif (inds != 1 and self.hurdle.positions + 1 < len(self.hurdle.map) and actions[inds] + 1 <= len(self.hurdle.map) and 
					self.hurdle.map[self.hurdle.positions + 1:actions[inds] + 1].count('#') == 0) or \
				(inds == 1 and actions[inds] < len(self.hurdle.map) and self.hurdle.map[actions[inds]] != '#'):

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
					self.scoreh -= 0

					
			
		if self.hurdle.positions >= 30:
			self.hurdle.map = 'GAME_OVER'
			
		if self.arc.wind != 'GAME_OVER' and self.arc.depth < len(self.arc.wind):
				
			w = int(self.arc.wind[self.arc.depth])
			x, y = apply_wind_effect(self.arc.player[0], self.arc.player[1], w, w, inds)
			self.arc.player = [x, y]
			score += 20 - math.sqrt(self.arc.player[0]**2 + self.arc.player[1]**2)
			self.arc.depth += 1
		if self.arc.depth >= len(self.arc.wind):
			self.arc.wind = 'GAME_OVER'

						

		if  self.dive.depth < len(self.dive.gpu) and self.dive.gpu != 'GAME_OVER':
			indsd = self.dive.get_action(self.dive.depth)
			if inds == indsd:
				self.data[3].reg4 = self.data[3].reg4 + 1
				#score += 20
			else:
				self.data[3].reg4 = 0
			self.data[3].reg1 = (self.data[3].reg1 + self.data[3].reg4 ) 
			score += self.data[3].reg1
			self.dive.depth += 1
		if  self.dive.depth >= len(self.data[3].gpu):
			self.dive.gpu = 'GAME_OVER'

		if self.roller.gpu != 'GAME_OVER':
			self.roller.move(inds)

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
		self.roller = Roller(data[2].gpu)

		self.data = data
		self.scoreh = 0

	def Advance(self, inds)-> int:
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
				# Ensure actions[inds] is a valid index
				if posh == -1:
					self.hurdle.positions = actions[inds]
					ind = self.hurdle.choose_action()
					if ind == inds:
						score  += self.hurdle.positions	+100
						self.scoreh += 2
					else:
						score  += self.hurdle.positions
						self.scoreh += 1

				elif (inds != 1 and self.hurdle.positions + 1 < len(self.hurdle.map) and actions[inds] + 1 <= len(self.hurdle.map) and 
					self.hurdle.map[self.hurdle.positions + 1:actions[inds] + 1].count('#') == 0) or \
				(inds == 1 and actions[inds] < len(self.hurdle.map) and self.hurdle.map[actions[inds]] != '#'):

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
					self.scoreh -= 0
						
		if self.hurdle.positions >= 30:
			self.hurdle.map = 'GAME_OVER'

		if self.arc.wind != 'GAME_OVER' and self.arc.depth < len(self.arc.wind):
				
			w = int(self.arc.wind[self.arc.depth])
			x, y = apply_wind_effect(self.arc.player[0], self.arc.player[1], w, w, inds)
			self.arc.player = [x, y]
			score += 20 - math.sqrt(self.arc.player[0]**2 + self.arc.player[1]**2)
			self.arc.depth += 1
		if self.arc.depth >= len(self.arc.wind):
			self.arc.wind = 'GAME_OVER'

		

		if  self.dive.depth < len(self.dive.gpu) and self.dive.gpu != 'GAME_OVER':
			indsd = self.dive.get_action(self.dive.depth)
			if inds == indsd:
				self.data[3].reg5 = self.data[3].reg5 + 1
				#score += 20
			else:
				self.data[3].reg5 = 0
			self.data[3].reg2 = self.data[3].reg2 + self.data[3].reg5
			score += self.data[3].reg2 
			self.dive.depth += 1
		if  self.dive.depth >= len(self.data[3].gpu):
			self.dive.gpu = 'GAME_OVER'

		if self.roller.gpu != 'GAME_OVER':
			self.roller.move(inds)

		return score

	def get_hurdle_score(self):
		return self.scoreh;

	def get_arc_score(self):
		return 20 - math.sqrt(self.arc.player[0]**2 + self.arc.player[1]**2)

	def get_dive_score(self):
		return self.data[3].reg2 

def calculate_mini_game_score(sorted_results):
    medals = {0: (0, 0), 1: (0, 0), 2: (0, 0)}  # {player_index: (gold, silver)}

    # Vérifier si plusieurs joueurs sont ex aequo à la première place
    if len(sorted_results) > 0:
        first_place_score = sorted_results[0][0]
        first_place_players = [player for score, player in sorted_results if score == first_place_score]

        # Si plusieurs joueurs partagent la première place, ils reçoivent tous une médaille d'or
        if len(first_place_players) > 1:
            for player in first_place_players:
                medals[player] = (1, 0)
        else:
            # Sinon, seul le premier reçoit une médaille d'or
            medals[first_place_players[0]] = (1, 0)

        # Vérifier s'il y a une deuxième place distincte pour les médailles d'argent
        if len(first_place_players) < len(sorted_results):
            second_place_score = sorted_results[len(first_place_players)][0]
            second_place_players = [player for score, player in sorted_results if score == second_place_score]

            # Attribuer une médaille d'argent à chaque joueur ex æquo pour la deuxième place
            for player in second_place_players:
                medals[player] = (0, 1)

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

	maxscore = 0
	minscore = 0

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
			
			#if depth2 < 5:
			if len(_node_p1.child) == 0:
				expandn(_node_p1, g, depth2, nbh, nba, nbd)
				"""game = choose_game(g.hurdle.stun_timers)
	
				
				if game ==  0 and g.hurdle.stun_timers == 0 and g.hurdle.map != 'GAME_OVER':
					
					inds = g.hurdle.choose_action()
				
								
				elif game == 1 and depth2 < len(data[3].gpu) and g.dive.gpu != 'GAME_OVER':
					inds = g.dive.get_action(depth2)
				
				else:
				inds = random.randint(0, 3)
				
				node_p1 = _node_p1.child[inds]
				node_p1.n += 1"""
			
				#print('obj=', _node_p1.child[inds], node_p1, file=sys.stderr, flush=True)
				
			#else:

			"""game = choose_game2(g.hurdle.stun_timers)
			if game ==  0 and g.hurdle.stun_timers == 0 and g.hurdle.map != 'GAME_OVER':
				
				inds = g.hurdle.choose_action()
			
							
			elif game == 1 and depth2 < len(data[3].gpu) and g.dive.gpu != 'GAME_OVER':
				inds = g.dive.get_action(depth2)
			
			else:"""
			inds = selection(_node_p1, node_p1)
			node_p1 = _node_p1.child[inds]	
			#node_p1.n += 1

			"""else:
				game = choose_game(g.hurdle.stun_timers)
							
				if game ==  0 and g.hurdle.stun_timers == 0 and g.hurdle.map != 'GAME_OVER':
					
					inds = g.hurdle.choose_action()
				
								
				elif game == 1 and depth2 < len(data[3].gpu) and g.dive.gpu != 'GAME_OVER':
					inds = g.dive.get_action(depth2)
				
				else:
					inds = random.randint(0, 3)"""
												
			#if depth2 < 5:

			if len(_node_p2.child) == 0:
				expandn(_node_p2, g2, depth2, nbh2, nba2, nbd2)
				"""
				game = choose_game(g2.hurdle.stun_timers)
				
				if game ==  0 and g2.hurdle.stun_timers == 0 and  g2.hurdle.map != 'GAME_OVER':
					
					inds2 = g2.hurdle.choose_action()
					
								
				elif game == 1 and depth2 < len(data[3].gpu) and g2.dive.gpu != 'GAME_OVER':
					inds2 = g2.dive.get_action(depth2)
				
				else:
					inds2 = random.randint(0, 3)
					
				node_p2 = _node_p2.child[inds2]
				node_p2.n += 1
			
			else:"""
			inds2 = selection(_node_p2, node_p2)
			node_p2 = _node_p2.child[inds2]		
			#node_p2.n += 1
			
			"""else:
				game = choose_game(g2.hurdle.stun_timers)
					
				if game ==  0 and g2.hurdle.stun_timers == 0 and  g2.hurdle.map != 'GAME_OVER':
					
					inds2 = g2.hurdle.choose_action()
					
								
				elif game == 1 and depth2 < len(data[3].gpu) and g2.dive.gpu != 'GAME_OVER':
					inds2 = g2.dive.get_action(depth2)
				
				else:
					inds2 = random.randint(0, 3)"""
				
			
			#if depth2 < 5:
			if len(_node_p3.child) == 0:
				expandn(_node_p3, g3, depth2, nbh3, nba3, nbd3)
				"""game = choose_game(g3.hurdle.stun_timers)
				
				if game == 0 and g3.hurdle.stun_timers == 0 and g3.hurdle.map != 'GAME_OVER':
					
					inds3 = g3.hurdle.choose_action()
				
								
				elif game == 1 and depth2 < len(data[3].gpu) and g3.dive.gpu != 'GAME_OVER':
					inds3 = g3.dive.get_action(depth2)
					
				else:
					inds3 = random.randint(0, 3)
					
				node_p3 = _node_p3.child[inds3]
				node_p3.n += 1
			
			else:"""
			inds3 = selection(_node_p3, node_p3)
			node_p3 = _node_p3.child[inds3]		
			#node_p3.n += 1

			"""else:
				game = choose_game(g3.hurdle.stun_timers)
					
				if game == 0 and g3.hurdle.stun_timers == 0 and g3.hurdle.map != 'GAME_OVER':
					
					inds3 = g3.hurdle.choose_action()
				
								
				elif game == 1 and depth2 < len(data[3].gpu) and g3.dive.gpu != 'GAME_OVER':
					inds3 = g3.dive.get_action(depth2)
					
				else:
					inds3 = random.randint(0, 3)"""
			
			ind_depth = 0
			#while 1:
			s1 += g.Advance(inds, depth2)
			s2 += g2.Advance(inds2, depth2)
			s3 += g3.Advance(inds3, depth2)

			ind_depth += 1

			if (g.hurdle.map == 'GAME_OVER' or g2.hurdle.map == 'GAME_OVER' or g3.hurdle.map == 'GAME_OVER') and \
			(g.arc.wind == 'GAME_OVER' or g2.arc.wind  == 'GAME_OVER' or g3.arc.wind  == 'GAME_OVER') and \
			(g.dive.gpu == 'GAME_OVER' or g2.dive.gpu  == 'GAME_OVER' or g3.dive.gpu  == 'GAME_OVER'):
				break
					
				
			#print('end rollout')
				
			#if depth2 < 5:
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
				
				
		for i in range(3):
			scoref[i] += player_scores[i]

		

		ch = 1.0
		ca = 1.0
		cd = 1.0

		if nbh == 0:ch = 5.0
		if nba == 0:ca = 3.0
		if nbd == 0:cd = 2.5

		bonus_h = ((g.get_hurdle_score() - g2.get_hurdle_score()) + (g.get_hurdle_score() - g3.get_hurdle_score()))
		bonus_a = ((g.get_arc_score() - g2.get_arc_score()) + (g.get_arc_score() - g3.get_arc_score()))
		bonus_d = ((g.get_dive_score() - g2.get_dive_score()) + (g.get_dive_score() - g3.get_dive_score()))
		sp1 = (g.get_hurdle_score()*ch+bonus_h) + (g.get_arc_score()*ca+bonus_a) + (g.get_dive_score()*cd + bonus_d)

		bonus_h = ((g2.get_hurdle_score() - g.get_hurdle_score()) + (g2.get_hurdle_score() - g3.get_hurdle_score()))
		bonus_a = ((g2.get_arc_score() - g.get_arc_score()) + (g2.get_arc_score() - g3.get_arc_score()))*2
		bonus_d = ((g2.get_dive_score() - g.get_dive_score()) + (g2.get_dive_score() - g3.get_dive_score()))*1.5
		sp2 = (g2.get_hurdle_score()+bonus_h) + (g2.get_arc_score()+bonus_a) + (g2.get_dive_score() + bonus_d)


		bonus_h = ((g3.get_hurdle_score() - g.get_hurdle_score()) + (g3.get_hurdle_score() - g2.get_hurdle_score()))
		bonus_a = ((g3.get_arc_score() - g.get_arc_score()) + (g3.get_arc_score() - g2.get_arc_score()))*2
		bonus_d = ((g3.get_dive_score() - g.get_dive_score()) + (g3.get_dive_score() - g2.get_dive_score()))*1.5
		sp3 = (g3.get_hurdle_score()+bonus_h) + (g3.get_arc_score()+bonus_a) + (g3.get_dive_score() + bonus_d)

	
		#sp1 += player_scores[player_idx]**2
		#sp2 += player_scores[player_ido[0]]**2
		#sp3 += player_scores[player_ido[1]]**2

		s1, s2, s3 = sp1, sp2, sp3
	
		sp1 += (scoref[player_idx]) 
		sp2 += (scoref[player_ido[0]])
		sp3 += (scoref[player_ido[1]])

		maxscore = max(maxscore, sp1)
		minscore = max(minscore, sp1)


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

		score = root.child[i].score / root.child[i].n

		print(i, score, file=sys.stderr, flush=True)

		if score > maxscore:
			maxscore = score
			indc = i

	res = root.child[indc].num
			

	return res

def expand_(node, g, num_game, epsilon):
	ind_n =-1
	
	# Epsilon-greedy selection
	if random.random() < epsilon:
		# Exploration: randomly choose a child
		game = choose_game(g.hurdle.stun_timers)
		
		if game ==  0 and g.hurdle.stun_timers == 0 and g.hurdle.map != 'GAME_OVER':
			
			ind_n = g.hurdle.choose_action()
					
						
		elif game == 1 and g.dive.gpu != 'GAME_OVER':
			ind_n = g.dive.get_action(g.dive.depth)
			
		else:
			ind_n = random.randint(0, 3)
	else:
		# Exploitation: choose the child with the maximum average score
		max_stat = -float('inf')
		ind_n = -1
		for i in range(len(node.child)):
			if node.child[i].n == 0:stat = MAX_DOUBLE
			else: stat = node.child[i].score / node.child[i].n
			if stat > max_stat:
				max_stat = stat
				ind_n = i

			
	return 1, ind_n
	

def expand_duct(root, root2, root3, g, g2, g3, player_scores,nbh, nba, nbd,num_game,num_game2, num_game3,player_id, player_ido,endt,  epsilon=0.2):
	depth = 0
	n = TNode()
	n2 = TNode()
	n3 = TNode()
	ind_n = ind_n2 = ind_n3 = 0
	f1 = f2 = f3 = 0
	sp1, sp2, sp3 = 0,0,0

	node = root
	node2 = root2
	node3 = root3

	leaf = None
	leaf2 = None
	leaf3 = None

	state_h = state_a = state_r = state_d = False
	state_f = False

	# List of tuples with initial values and names for reference
	values = [('nbh', nbh), ('nba', nba), ('nbd', nbd)]

	# Sort the values in ascending order
	values.sort(key=lambda x: x[1])

	# Assign scores based on sorted order
	scores = {values[0][0]: 10, values[1][0]: 2, values[2][0]: 1}

	# Assign the score values to ch, ca, cd based on the variable names
	ch = scores['nbh']
	ca = scores['nba']
	cd = scores['nbd']

	enter1 = enter2 = enter3 = 0

	#print('in=',timedelta(seconds=time.perf_counter()), timedelta(seconds=endt), file=sys.stderr, flush=True)

	depth = 0
	while time.perf_counter() < endt-0.002:
		if f1 == 0:
			r, r1 = expand_(node, g, num_game, epsilon)
			if node.child[r1].create == False:
				node.child[r1] = TNode()
				node.child[r1].num = r1
				node.child[r1].n = 0
				node.child[r1].parent = node
				enter1 = 1
				leaf = node.child[r1]
				g.Advance(r1)
				f1 = 1
			else:
				node = node.child[r1]
				g.Advance(r1)
		else:
			ind_n = GetRandom(g, num_game)
			g.Advance(ind_n)
		


		if f2 == 0:
			r, r1 = expand_(node2, g2,num_game2, epsilon)
			
			if node2.child[r1].create == False:
				node2.child[r1] = TNode()
				node2.child[r1].num = r1
				node2.child[r1].n = 0
				node2.child[r1].parent = node2
				leaf2 = node2.child[r1]
				enter2 = 1
				g2.Advance(r1)
				f2 = 1
			
			else:
				node2 = node2.child[r1]
				g2.Advance(r1)
		else:
			ind_n2 = GetRandom(g2, num_game2)
			g2.Advance(ind_n2)

		

		if f3 == 0:
			r, r1 = expand_(node3, g3,num_game3, epsilon)
			if node3.child[r1].create == False:
				node3.child[r1] = TNode()
				node3.child[r1].num = r1
				node3.child[r1].n = 0
				node3.child[r1].parent = node3
				enter3 = 1
				leaf3 = node3.child[r1]
				g3.Advance(r1)
				f3 = 1
			else:
				node3 = node3.child[r1]
				g3.Advance(r1)
		else:
			ind_n3 =GetRandom(g3, num_game3)
			g3.Advance(ind_n3)
			
		mask =[0] * 3

				
		#if f1 == 0 or enter1 == 1:
		bonus_h = ((g.get_hurdle_score() - g2.get_hurdle_score()) + (g.get_hurdle_score() - g3.get_hurdle_score())) / 15.0
		bonus_a = ((g.get_arc_score() - g2.get_arc_score()) + (g.get_arc_score() - g3.get_arc_score())) / 20.0
		bonus_d = ((g.get_dive_score() - g2.get_dive_score()) + (g.get_dive_score() - g3.get_dive_score())) / 50.0
		bonus_r = ((g.roller.position - g2.roller.position) + (g.roller.position - g3.roller.position)) / 50.0
		
		sp1 += g.get_hurdle_score()/30.0*ch+bonus_h
		sp1 += g.roller.position / 150.0 + bonus_r
		sp1 +=g.get_arc_score()/20.0*ca+bonus_a
		
		sp1 +=g.get_dive_score()/50.0*cd+bonus_d
		enter1 = 2
		mask[player_id] = 1

		#if f2 == 0 or enter2 ==1 :
		bonus_h = ((g2.get_hurdle_score() - g.get_hurdle_score()) + (g2.get_hurdle_score() - g3.get_hurdle_score())) / 15.0
		bonus_a = ((g2.get_arc_score() - g.get_arc_score()) + (g2.get_arc_score() - g3.get_arc_score())) / 20.0
		bonus_d = ((g2.get_dive_score() - g.get_dive_score()) + (g2.get_dive_score() - g3.get_dive_score())) / 50.0
		bonus_r = ((g2.roller.position - g.roller.position) + (g2.roller.position - g3.roller.position)) / 50.0

		sp2 += g2.get_hurdle_score()/30.0 + bonus_h
		sp2 += g2.roller.position / 150.0 + bonus_r
		sp2 += g2.get_arc_score()/20.0 + bonus_a
		sp2 += g2.get_dive_score()/50.0 + bonus_d
		enter2 = 2
		mask[player_ido[0]] = 1

		#if f3 == 0 or enter3 == 1:
		bonus_h = ((g3.get_hurdle_score() - g.get_hurdle_score()) + (g3.get_hurdle_score() - g2.get_hurdle_score())) / 15.0
		bonus_a = ((g3.get_arc_score() - g.get_arc_score()) + (g3.get_arc_score() - g2.get_arc_score())) / 20.0
		bonus_d = ((g3.get_dive_score() - g.get_dive_score()) + (g3.get_dive_score() - g2.get_dive_score())) / 50.0
		bonus_r = ((g3.roller.position - g.roller.position) + (g3.roller.position - g2.roller.position)) / 50.0

		sp3 += g3.get_hurdle_score()/30.0 + bonus_h
		sp3 += g3.roller.position / 150.0 + bonus_r
		sp3 += g3.get_arc_score()/20.0 + bonus_a
		sp3 += g3.get_dive_score()/50.0 + bonus_d
		enter3 = 2
		mask[player_ido[1]] = 1
		
			
		if (g.hurdle.map == 'GAME_OVER' or g2.hurdle.map == 'GAME_OVER' or g3.hurdle.map == 'GAME_OVER'):
		#if f1 == 0 and f2 == 0 and f3 == 0:

			mh = sorted([(g.hurdle.positions, player_id), (g2.hurdle.positions, player_ido[0]), (g3.hurdle.positions, player_ido[1])], key=lambda x: x[0], reverse=True)

			state_h = True
			"""hur = Create_Hurdle()
			g.hurdle = hur.clone()
			g2.hurdle = hur.clone()
			g3.hurdle = hur.clone()"""
					
			
			"""mini_game_score = calculate_mini_game_score(mh)
			for player, score in mini_game_score.items():
				if mask[player] == 1:
					player_scores[player] *= score"""
			

		if (g.roller.gpu == 'GAME_OVER' or g2.roller.gpu == 'GAME_OVER' or g3.roller.gpu == 'GAME_OVER'):
		#if f1 == 0 and f2 == 0 and f3 == 0:

			mr = sorted([(g.roller.position, player_id), (g2.roller.position, player_ido[0]), (g3.roller.position, player_ido[1])], key=lambda x: x[0], reverse=True)

			state_r = True
			"""g.roller.reset()
			g2.roller.reset()
			g3.roller.reset()"""
		

		if (g.arc.wind == 'GAME_OVER' or g2.arc.wind  == 'GAME_OVER' or g3.arc.wind  == 'GAME_OVER') :
		#if f1 == 0 and f2 == 0 and f3 == 0:

			ma = sorted([(g.arc.euclidean_distance(g.arc.player[0], g.arc.player[1]), player_id),
			(g2.arc.euclidean_distance(g2.arc.player[0], g2.arc.player[1]), player_ido[0]),
			(g3.arc.euclidean_distance(g3.arc.player[0], g3.arc.player[1]), player_ido[1])],  key=lambda x: x[0], reverse=True)

			state_a = True

			"""arc = Create_arc()
			g.arc = arc.clone()
			g2.arc = arc.clone()
			g3.arc = arc.clone()"""
	
			
			"""mini_game_score = calculate_mini_game_score(ma)
			for player, score in mini_game_score.items():
				if mask[player] == 1:
					player_scores[player] *= score"""
			

		if (g.dive.gpu == 'GAME_OVER' or g2.dive.gpu  == 'GAME_OVER' or g3.dive.gpu  == 'GAME_OVER'):
		#if f1 == 0 and f2 == 0 and f3 == 0:
			md = sorted([(g.get_dive_score(), player_id), (g2.get_dive_score(), player_ido[0]), (g3.get_dive_score(), player_ido[1])],  key=lambda x: x[0], reverse=True)
		
			state_d = True

			"""d = Create_Dive()
			g.dive = d.clone()
			g2.dive = d.clone()
			g3.dive = d.clone()"""
	
			
			"""mini_game_score = calculate_mini_game_score(md)
			for player, score in mini_game_score.items():
				if mask[player] == 1:
					player_scores[player] *= score"""
			
		if state_a and state_h and state_r and state_d:
			if leaf == None:leaf = node
			if leaf2 == None:leaf2 = node2
			if leaf3 == None:leaf3 = node3
			state_f = True
			break
		if f1 == 1 and f2 == 1 and f3 == 1:break
		depth += 1	

	#print('out=', timedelta(seconds=time.perf_counter()), timedelta(seconds=endt), file=sys.stderr, flush=True)
	

	# Create and add a new node as a child to the current node
	

	return leaf, leaf2, leaf3,sp1, sp2, sp3, depth,state_f

def GetRandom(g, num_game):
	ind_n = -1
	game = choose_game(g.hurdle.stun_timers)
			
	if game ==  0 and g.hurdle.stun_timers == 0 and g.hurdle.map != 'GAME_OVER':
		
		ind_n = g.hurdle.choose_action()
				
					
	elif game == 1 and g.dive.gpu != 'GAME_OVER':
		ind_n = g.dive.get_action(g.dive.depth)
		
	else:
		ind_n = random.randint(0, 3)

	return ind_n

def rollout(nb_turn, g, g2, g3, player_scores, player_id, player_ido, num_game, num_game2, num_game3):

	sp1, sp2, sp3 = 0, 0, 0

	state_a = state_h = state_d = state_r = False

	for i in range(nb_turn):

		ind_n = GetRandom(g, num_game)
		ind_n2 = GetRandom(g2, num_game2)
		ind_n3 = GetRandom(g3, num_game3)

		g.Advance(ind_n)
		g2.Advance(ind_n2)
		g3.Advance(ind_n3)

		"""
		mh = sorted([(g.hurdle.positions, player_id), (g2.hurdle.positions, player_ido[0]), (g3.hurdle.positions, player_ido[1])], key=lambda x: x[0], reverse=True)
		mr = sorted([(g.roller.position, player_id), (g2.roller.position, player_ido[0]), (g3.roller.position, player_ido[1])], key=lambda x: x[0], reverse=True)
		ma = sorted([(g.arc.euclidean_distance(g.arc.player[0], g.arc.player[1]), player_id),
			(g2.arc.euclidean_distance(g2.arc.player[0], g2.arc.player[1]), player_ido[0]),
			(g3.arc.euclidean_distance(g3.arc.player[0], g3.arc.player[1]), player_ido[1])],  key=lambda x: x[0], reverse=True)
		md = sorted([(g.get_dive_score(), player_id), (g2.get_dive_score(), player_ido[0]), (g3.get_dive_score(), player_ido[1])],  key=lambda x: x[0], reverse=True)
		
		rec = [3, 1, 0.1]
		i = 0
		for p, idp in mh:
			if idp == player_id:
				sp1 +=rec[i]
			elif idp == player_ido[0]:
				sp2 += rec[i]
			elif idp == player_ido[1]:
				sp3 += rec[i]

			i+=1

		i = 0
		for p, idp in mr:
			if idp == player_id:
				sp1 +=rec[i]
			elif idp == player_ido[0]:
				sp2 += rec[i]
			elif idp == player_ido[1]:
				sp3 += rec[i]

			i+=1

		i = 0
		for p, idp in ma:
			if idp == player_id:
				sp1 +=rec[i]
			elif idp == player_ido[0]:
				sp2 += rec[i]
			elif idp == player_ido[1]:
				sp3 += rec[i]

			i+=1

		i = 0
		for p, idp in md:
			if idp == player_id:
				sp1 +=rec[i]
			elif idp == player_ido[0]:
				sp2 += rec[i]
			elif idp == player_ido[1]:
				sp3 += rec[i]

			i+=1
		"""

		
		bonus_h = ((g.get_hurdle_score() - g2.get_hurdle_score()) + (g.get_hurdle_score() - g3.get_hurdle_score())) / 15.0
		bonus_a = ((g.get_arc_score() - g2.get_arc_score()) + (g.get_arc_score() - g3.get_arc_score())) / 20.0
		bonus_d = ((g.get_dive_score() - g2.get_dive_score()) + (g.get_dive_score() - g3.get_dive_score())) / 50.0
		bonus_r = ((g.roller.position - g2.roller.position) + (g.roller.position - g3.roller.position)) / 50.0
		
		sp1 += g.get_hurdle_score()/30.0+bonus_h 
		sp1 += g.roller.position / 150.0 + bonus_r
		sp1 += g.get_arc_score()/20.0+bonus_a
		
		sp1 += g.get_dive_score()/50.0+bonus_d
		



		bonus_h = ((g2.get_hurdle_score() - g.get_hurdle_score()) + (g2.get_hurdle_score() - g3.get_hurdle_score())) / 15.0
		bonus_a = ((g2.get_arc_score() - g.get_arc_score()) + (g2.get_arc_score() - g3.get_arc_score())) / 20.0
		bonus_d = ((g2.get_dive_score() - g.get_dive_score()) + (g2.get_dive_score() - g3.get_dive_score())) / 50.0
		bonus_r = ((g2.roller.position - g.roller.position) + (g2.roller.position - g3.roller.position)) / 50.0

		sp2 += g2.get_hurdle_score()/30.0 + bonus_h
		sp2 += g2.roller.position / 150.0 + bonus_r
		sp2 += g2.get_arc_score()/20.0+ bonus_a
		sp2 += g2.get_dive_score()/50.0 + bonus_d
		


		bonus_h = ((g3.get_hurdle_score() - g.get_hurdle_score()) + (g3.get_hurdle_score() - g2.get_hurdle_score())) / 15.0
		bonus_a = ((g3.get_arc_score() - g.get_arc_score()) + (g3.get_arc_score() - g2.get_arc_score())) / 20.0
		bonus_d = ((g3.get_dive_score() - g.get_dive_score()) + (g3.get_dive_score() - g2.get_dive_score())) / 50.0
		bonus_r = ((g3.roller.position - g.roller.position) + (g3.roller.position - g2.roller.position)) / 50.0

		sp3 += g3.get_hurdle_score()/30.0 + bonus_h
		sp3 += g3.roller.position / 150.0 + bonus_r
		sp3 += g3.get_arc_score()/20.0 + bonus_a
		sp3 += g3.get_dive_score()/50.0+ bonus_d
		

			
		if (g.hurdle.map == 'GAME_OVER' or g2.hurdle.map == 'GAME_OVER' or g3.hurdle.map == 'GAME_OVER'):
		#if f1 == 0 and f2 == 0 and f3 == 0:

			mh = sorted([(g.hurdle.positions, player_id), (g2.hurdle.positions, player_ido[0]), (g3.hurdle.positions, player_ido[1])], key=lambda x: x[0], reverse=True)

			"""hur = Create_Hurdle()
			g.hurdle = hur.clone()
			g2.hurdle = hur.clone()
			g3.hurdle = hur.clone()"""

			state_h = True
					
			
			mini_game_score = calculate_mini_game_score(mh)
			for player, score in mini_game_score.items():
				player_scores[player] *= score
			

		if (g.roller.gpu == 'GAME_OVER' or g2.roller.gpu == 'GAME_OVER' or g3.roller.gpu == 'GAME_OVER'):
		#if f1 == 0 and f2 == 0 and f3 == 0:

			mr = sorted([(max(150, g.roller.position), player_id), (max(150, g2.roller.position), player_ido[0]), (max(150,g3.roller.position), player_ido[1])], key=lambda x: x[0], reverse=True)

			"""g.roller.reset()
			g2.roller.reset()
			g3.roller.reset()"""

			state_r = True

			mini_game_score = calculate_mini_game_score(mr)
			for player, score in mini_game_score.items():
				player_scores[player] *= score


		if (g.arc.wind == 'GAME_OVER' or g2.arc.wind  == 'GAME_OVER' or g3.arc.wind  == 'GAME_OVER') :
		#if f1 == 0 and f2 == 0 and f3 == 0:

			ma = sorted([(g.arc.euclidean_distance(g.arc.player[0], g.arc.player[1]), player_id),
			(g2.arc.euclidean_distance(g2.arc.player[0], g2.arc.player[1]), player_ido[0]),
			(g3.arc.euclidean_distance(g3.arc.player[0], g3.arc.player[1]), player_ido[1])],  key=lambda x: x[0], reverse=True)

			state_a = True

			"""arc = Create_arc()
			g.arc = arc.clone()
			g2.arc = arc.clone()
			g3.arc = arc.clone()"""

			
			mini_game_score = calculate_mini_game_score(ma)
			for player, score in mini_game_score.items():
				player_scores[player] *= score
			

		if (g.dive.gpu == 'GAME_OVER' or g2.dive.gpu  == 'GAME_OVER' or g3.dive.gpu  == 'GAME_OVER'):
		#if f1 == 0 and f2 == 0 and f3 == 0:
			md = sorted([(g.get_dive_score(), player_id), (g2.get_dive_score(), player_ido[0]), (g3.get_dive_score(), player_ido[1])],  key=lambda x: x[0], reverse=True)
		
			"""d = Create_Dive()
			g.dive = d.clone()
			g2.dive = d.clone()
			g3.dive = d.clone()"""

			state_d = True

			
			mini_game_score = calculate_mini_game_score(md)
			for player, score in mini_game_score.items():
				player_scores[player] *= score

		if state_a and state_h and state_r and state_d:
			
			break


	return sp1, sp2, sp3
	
def duct(depth, data, ind, medals, player_id, player_ido, gld)-> int:

	root = TNode()
	root.parent = None
	root.score = 0

	root2 = TNode()
	root2.parent = None
	root2.score = 0

	root3 = TNode()
	root3.parent = None
	root3.score = 0

	g = Game()
	g2 = Game2()
	g3 = Game3()

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

	mult = 10

	nbh = medals[player_id][1] #* mult + medals[player_id][2]
	nba = medals[player_id][4] #* mult + medals[player_id][5]
	nbd = medals[player_id][10] #* mult + medals[player_id][11]

	nbh2 = medals[player_ido[0]][1] #* mult + medals[player_ido[0]][2]
	nba2 = medals[player_ido[0]][4] #* mult + medals[player_ido[0]][5]
	nbd2 = medals[player_ido[0]][10] #* mult + medals[player_ido[0]][11]

	nbh3 = medals[player_ido[1]][1] #* mult + medals[player_ido[1]][2]
	nba3 = medals[player_ido[1]][4] #* mult + medals[player_ido[1]][5]
	nbd3 = medals[player_ido[1]][10] #* mult + medals[player_ido[1]][11]


	end = time.perf_counter() + 0.05

	difft = 0

	turn = 0

	max_depth = 0

	# Store in a list to easily find the index
	values = [nbh, nba, nbd]

	# Get the index of the maximum value
	num_game = values.index(min(values))

	values = [nbh2, nba2, nbd2]

	# Get the index of the maximum value
	num_game2 = values.index(min(values))

	values = [nbh3, nba3, nbd3]

	# Get the index of the maximum value
	num_game3 = values.index(min(values))


	"""
	if num_game ==  0 and g.hurdle.stun_timers == 0 and g.hurdle.map != 'GAME_OVER':
				
		return g.hurdle.choose_action()
				
					
	elif num_game == 2 and g.dive.gpu != 'GAME_OVER':
		return g.dive.get_action(g.dive.depth)"""
			
	eps = 0.25
	while time.perf_counter()+difft < end:
		start = time.perf_counter()
		player_scores = {0: 1, 1: 1, 2: 1}

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
		
		depth = 0
		sp1, sp2, sp3 = 0, 0, 0

		
		leaf, leaf2, leaf3, sp1, sp2, sp3, depth, state_f  = expand_duct(root, root2, root3, g, g2, g3, player_scores,nbh, nba, nbd,num_game,num_game2, num_game3,player_id, player_ido,end, eps)
		
		
		s1, s2, s3 = rollout(2, g, g2, g3, player_scores, player_id, player_ido, num_game, num_game2, num_game3)

		#b1 = (player_scores[player_id] - player_scores[player_ido[0]]) +  (player_scores[player_id] - player_scores[player_ido[1]])
		#b2 = (player_scores[player_ido[0]] - player_scores[player_id]) +  (player_scores[player_ido[0]] - player_scores[player_ido[1]])
		#b3 = (player_scores[player_ido[1]] - player_scores[player_id]) +  (player_scores[player_ido[1]] - player_scores[player_ido[0]])
		reward1 = sp1 +s1/5.0
		reward2 = sp2+s2/5.0 
		reward3 = sp3+s3/5.0 

		totr = reward1 + reward2 + reward3
		reward1 /= totr
		reward2 /= totr
		reward3 /= totr

		max_depth = max(max_depth, depth)

		backpropagation(leaf, reward1)
		backpropagation(leaf2, reward2)
		backpropagation(leaf3, reward3)

		endt = time.perf_counter()
		difft = max(difft, endt - start)

		turn += 1

		#if turn % 10 == 0:
			
	print('turn=', turn, max_depth, file=sys.stderr, flush=True)

	indc = -1
	maxscore = -MAX_DOUBLE
	for i in range(len(root.child)):
		score = 0

		if root.child[i].n == 0:
			continue

		score = root.child[i].score / root.child[i].n

		print(i, score, root.child[i].n, file=sys.stderr, flush=True)

		if score > maxscore:
			maxscore = score
			indc = i

	res = indc
			

	return res


def find_leaves(node):
        leaves = []
        ind = []
        for i, sublist1 in enumerate(node.child64):
            for j, sublist2 in enumerate(sublist1):
                for k, value in enumerate(sublist2):
                    if value is not None:
                        leaves.append((i, j, k, value))
                    else:
                        ind.append((i, j, k))
        return leaves, ind

def find_move(node, g, ind_p, epsilon):

	ind_n =-1
	# Epsilon-greedy selection
	if random.random() < epsilon:
		# Exploration: randomly choose a child
		game = choose_game(g.hurdle.stun_timers)
		
		if game ==  0 and g.hurdle.stun_timers == 0 and g.hurdle.map != 'GAME_OVER':
			
			ind_n = g.hurdle.choose_action()
					
						
		elif game == 1 and g.dive.gpu != 'GAME_OVER':
			ind_n = g.dive.get_action(g.dive.depth)
			
		else:
			ind_n = random.randint(0, 3)
	else:
		# Exploitation: choose the child with the maximum average score
		max_stat = -float('inf')
		ind_n = -1
		for i in range(len(node.childp[ind_p])):
			n = node.childp[ind_p][i].n
			if n == 0: n = 1
			stat = node.childp[ind_p][i].score / n
			if stat > max_stat:
				max_stat = stat
				ind_n = i

			
	return 1, ind_n
	

def backpropagation64(parent, sc, sc2, sc3):
	par = parent
	last = parent	
	i = 0
	while par is not None:
		#print('turn=', par.childp, len(par.childp[0]),par.ind_p1, file=sys.stderr, flush=True)
		if i > 0:
			par.childp[0][last.ind_p1].n += 1
			par.childp[0][last.ind_p1].score += sc


			par.childp[1][last.ind_p2].n += 1
			par.childp[1][last.ind_p2].score += sc2


			par.childp[2][last.ind_p3].n += 1
			par.childp[2][last.ind_p3].score += sc3

		i += 1
		last = par
		par = par.parent


def expand_duct64(root, g, g2, g3, player_scores,nbh, nba, nbd,num_game,num_game2, num_game3,player_id, player_ido, epsilon=0.2):
	depth = 0
	n = TNode()
	n2 = TNode()
	n3 = TNode()
	ind_n = ind_n2 = ind_n3 = 0
	f1 = f2 = f3 = 0
	sp1, sp2, sp3 = 0,0,0

	node = root
	
	leaf = None
	leaf2 = None
	leaf3 = None

	# List of tuples with initial values and names for reference
	values = [('nbh', nbh), ('nba', nba), ('nbd', nbd)]

	# Sort the values in ascending order
	values.sort(key=lambda x: x[1])

	# Assign scores based on sorted order
	scores = {values[0][0]: 10, values[1][0]: 2, values[2][0]: 1}

	# Assign the score values to ch, ca, cd based on the variable names
	ch = scores['nbh']
	ca = scores['nba']
	cd = scores['nbd']

	enter1 = enter2 = enter3 = 0

	G = [g, g2, g3]
	depth = 0
	while 1:
		ind_n = [0] * 3
		for ind_p in range(3):
			r, _ind_n, = find_move(node, G[ind_p], ind_p, epsilon)
			ind_n[ind_p] = _ind_n
			
		#print('turn=', ind_n, file=sys.stderr, flush=True)
		if node.child64[ind_n[0]][ind_n[1]][ind_n[2]] == None:
	
			n = TNode()
			n.n = 1
			n.parent = node
			n.ind_p1 = ind_n[0]
			n.ind_p2 = ind_n[1]
			n.ind_p3 = ind_n[2]
							
			node.child64[ind_n[0]][ind_n[1]][ind_n[2]] = n
			leaf = n
			g.Advance(ind_n[0])
			g2.Advance(ind_n[1])
			g3.Advance(ind_n[2])
			f1 = 1		
			
		else:
			node = node.child64[ind_n[0]][ind_n[1]][ind_n[2]]
			g.Advance(ind_n[0])
			g2.Advance(ind_n[1])
			g3.Advance(ind_n[2])
		
		
			
		mask =[0] * 3

		#if f1 == 0 or enter1 == 1:
		bonus_h = ((g.get_hurdle_score() - g2.get_hurdle_score()) + (g.get_hurdle_score() - g3.get_hurdle_score())) / 15.0
		bonus_a = ((g.get_arc_score() - g2.get_arc_score()) + (g.get_arc_score() - g3.get_arc_score())) / 20.0
		bonus_d = ((g.get_dive_score() - g2.get_dive_score()) + (g.get_dive_score() - g3.get_dive_score())) / 50.0
		bonus_r = ((g.roller.position - g2.roller.position) + (g.roller.position - g3.roller.position)) / 50.0
		
		sp1 += g.get_hurdle_score()/30.0*ch+bonus_h
		sp1 += g.roller.position / 150.0 + bonus_r
		sp1 +=g.get_arc_score()/20.0*ca+bonus_a
		
		sp1 +=g.get_dive_score()/50.0*cd+bonus_d
		enter1 = 2
		mask[player_id] = 1

		#if f2 == 0 or enter2 ==1 :
		bonus_h = ((g2.get_hurdle_score() - g.get_hurdle_score()) + (g2.get_hurdle_score() - g3.get_hurdle_score())) / 15.0
		bonus_a = ((g2.get_arc_score() - g.get_arc_score()) + (g2.get_arc_score() - g3.get_arc_score())) / 20.0
		bonus_d = ((g2.get_dive_score() - g.get_dive_score()) + (g2.get_dive_score() - g3.get_dive_score())) / 50.0
		bonus_r = ((g2.roller.position - g.roller.position) + (g2.roller.position - g3.roller.position)) / 50.0

		sp2 += g2.get_hurdle_score()/30.0 + bonus_h
		sp2 += g2.roller.position / 150.0 + bonus_r
		sp2 += g2.get_arc_score()/20.0 + bonus_a
		sp2 += g2.get_dive_score()/50.0 + bonus_d
		enter2 = 2
		mask[player_ido[0]] = 1

		#if f3 == 0 or enter3 == 1:
		bonus_h = ((g3.get_hurdle_score() - g.get_hurdle_score()) + (g3.get_hurdle_score() - g2.get_hurdle_score())) / 15.0
		bonus_a = ((g3.get_arc_score() - g.get_arc_score()) + (g3.get_arc_score() - g2.get_arc_score())) / 20.0
		bonus_d = ((g3.get_dive_score() - g.get_dive_score()) + (g3.get_dive_score() - g2.get_dive_score())) / 50.0
		bonus_r = ((g3.roller.position - g.roller.position) + (g3.roller.position - g2.roller.position)) / 50.0

		sp3 += g3.get_hurdle_score()/30.0 + bonus_h
		sp3 += g3.roller.position / 150.0 + bonus_r
		sp3 += g3.get_arc_score()/20.0 + bonus_a
		sp3 += g3.get_dive_score()/50.0 + bonus_d
		enter3 = 2
		mask[player_ido[1]] = 1

			
		if (g.hurdle.map == 'GAME_OVER' or g2.hurdle.map == 'GAME_OVER' or g3.hurdle.map == 'GAME_OVER'):
		#if f1 == 0 and f2 == 0 and f3 == 0:

			mh = sorted([(g.hurdle.positions, player_id), (g2.hurdle.positions, player_ido[0]), (g3.hurdle.positions, player_ido[1])], key=lambda x: x[0], reverse=True)

			hur = Create_Hurdle()
			g.hurdle = hur.clone()
			g2.hurdle = hur.clone()
			g3.hurdle = hur.clone()
					
			
			"""mini_game_score = calculate_mini_game_score(mh)
			for player, score in mini_game_score.items():
				if mask[player] == 1:
					player_scores[player] *= score"""
			

		if (g.roller.gpu == 'GAME_OVER' or g2.roller.gpu == 'GAME_OVER' or g3.roller.gpu == 'GAME_OVER'):
		#if f1 == 0 and f2 == 0 and f3 == 0:

			mr = sorted([(g.roller.position, player_id), (g2.roller.position, player_ido[0]), (g3.roller.position, player_ido[1])], key=lambda x: x[0], reverse=True)

			g.roller.reset()
			g2.roller.reset()
			g3.roller.reset()
		

		if (g.arc.wind == 'GAME_OVER' or g2.arc.wind  == 'GAME_OVER' or g3.arc.wind  == 'GAME_OVER') :
		#if f1 == 0 and f2 == 0 and f3 == 0:

			ma = sorted([(g.arc.euclidean_distance(g.arc.player[0], g.arc.player[1]), player_id),
			(g2.arc.euclidean_distance(g2.arc.player[0], g2.arc.player[1]), player_ido[0]),
			(g3.arc.euclidean_distance(g3.arc.player[0], g3.arc.player[1]), player_ido[1])],  key=lambda x: x[0], reverse=True)

			arc = Create_arc()
			g.arc = arc.clone()
			g2.arc = arc.clone()
			g3.arc = arc.clone()
	
			
			"""mini_game_score = calculate_mini_game_score(ma)
			for player, score in mini_game_score.items():
				if mask[player] == 1:
					player_scores[player] *= score"""
			

		if (g.dive.gpu == 'GAME_OVER' or g2.dive.gpu  == 'GAME_OVER' or g3.dive.gpu  == 'GAME_OVER'):
		#if f1 == 0 and f2 == 0 and f3 == 0:
			md = sorted([(g.get_dive_score(), player_id), (g2.get_dive_score(), player_ido[0]), (g3.get_dive_score(), player_ido[1])],  key=lambda x: x[0], reverse=True)
		
			d = Create_Dive()
			g.dive = d.clone()
			g2.dive = d.clone()
			g3.dive = d.clone()
	
			
			"""mini_game_score = calculate_mini_game_score(md)
			for player, score in mini_game_score.items():
				if mask[player] == 1:
					player_scores[player] *= score"""
			

		if f1 == 1 :break
		depth += 1	

	# Create and add a new node as a child to the current node
	

	return leaf,sp1, sp2, sp3, depth


def duct64(depth, data, ind, medals, player_id, player_ido, gld)-> int:

	root = TNode()
	root.parent = None
	root.score = 0
	
	g = Game()
	g2 = Game2()
	g3 = Game3()

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

	mult = 3

	nbh = medals[player_id][1] * mult + medals[player_id][2]
	nba = medals[player_id][4] * mult + medals[player_id][5]
	nbd = medals[player_id][10] * mult + medals[player_id][11]

	nbh2 = medals[player_ido[0]][1] * mult + medals[player_ido[0]][2]
	nba2 = medals[player_ido[0]][4] * mult + medals[player_ido[0]][5]
	nbd2 = medals[player_ido[0]][10] * mult + medals[player_ido[0]][11]

	nbh3 = medals[player_ido[1]][1] * mult + medals[player_ido[1]][2]
	nba3 = medals[player_ido[1]][4] * mult + medals[player_ido[1]][5]
	nbd3 = medals[player_ido[1]][10] * mult + medals[player_ido[1]][11]


	end = time.perf_counter() + 0.040

	difft = 0

	turn = 0

	max_depth = 0

	# Store in a list to easily find the index
	values = [nbh, nba, nbd]

	# Get the index of the maximum value
	num_game = values.index(min(values))

	values = [nbh2, nba2, nbd2]

	# Get the index of the maximum value
	num_game2 = values.index(min(values))

	values = [nbh3, nba3, nbd3]

	# Get the index of the maximum value
	num_game3 = values.index(min(values))


	"""
	if num_game ==  0 and g.hurdle.stun_timers == 0 and g.hurdle.map != 'GAME_OVER':
				
		return g.hurdle.choose_action()
				
					
	elif num_game == 2 and g.dive.gpu != 'GAME_OVER':
		return g.dive.get_action(g.dive.depth)"""
			

	while time.perf_counter()+difft < end:
		start = time.perf_counter()
		player_scores = {0: 1, 1: 1, 2: 1}

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
		
		depth = 0
		sp1, sp2, sp3 = 0, 0, 0

		eps = 0.25
		leaf, sp1, sp2, sp3, depth  = expand_duct64(root, g, g2, g3, player_scores,nbh, nba, nbd,num_game,num_game2, num_game3,player_id, player_ido, eps)

		s1, s2, s3 = rollout(5, g, g2, g3, player_scores, player_id, player_ido, num_game, num_game2, num_game3)

		#b1 = (player_scores[player_id] - player_scores[player_ido[0]]) +  (player_scores[player_id] - player_scores[player_ido[1]])
		#b2 = (player_scores[player_ido[0]] - player_scores[player_id]) +  (player_scores[player_ido[0]] - player_scores[player_ido[1]])
		#b3 = (player_scores[player_ido[1]] - player_scores[player_id]) +  (player_scores[player_ido[1]] - player_scores[player_ido[0]])
		reward1 = sp1 + s1 / 30.0
		reward2 = sp2 + s2 / 30.0
		reward3 = sp3 + s3 / 30.0

		"""totr = reward1 + reward2 + reward3
		reward1 /= totr
		reward2 /= totr
		reward3 /= totr"""

		max_depth = max(max_depth, depth)

		backpropagation64(leaf, reward1, reward2, reward3)
		#backpropagation(leaf2, reward2)
		#backpropagation(leaf3, reward3)

		endt = time.perf_counter()
		difft = max(difft, endt - start)

		turn += 1


	print('turn=', turn, max_depth, file=sys.stderr, flush=True)

	indc = -1
	maxscore = -MAX_DOUBLE
	for i in range(4):
		

		if root.childp[0][i].n == 0:
			continue

		score = root.childp[0][i].score / root.childp[0][i].n

		print(i, score, root.childp[0][i].n, file=sys.stderr, flush=True)

		if score > maxscore:
			maxscore = score
			indc = i

	res = indc
			

	return res


def Prepare_Game(data, medals, player_ido) -> int:

	global nb_turn 
	global turn 
	global ind_game
	global player_idx
	
	print('player=', player_idx, file=sys.stderr, flush=True)

	gold = [(medals[player_idx][1], 0),(medals[player_idx][4], 1),(medals[player_idx][7], 2),(medals[player_idx][10], 3)]
	gold_sorted = sorted(gold, key=lambda x: x[0], reverse=False)
	
	m = 10
	gld = [1]*4
	#for g in gold_sorted:
	gld[gold_sorted[0][1]]= 100
	m = m / 2
		
	action_g = 0
	#max(5, 30-data[0].reg0)
	action_g = duct(5, data, 1, medals, player_idx, player_ido, gld)
	
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

    medals = []
    player_ido = []
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
    
    print(actions[action], flush=True)

    



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