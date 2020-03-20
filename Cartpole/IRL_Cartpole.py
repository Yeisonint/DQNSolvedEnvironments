# Librerias
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import cvxpy as cp # Necesaria para problema convexo

# Cartpole
import gym
import NNAgent

# Parametros del entorno
SAVEDIR = './Cartpole/Train_IRL/'
env = gym.make('CartPole-v1')
ACTION_SPACE = env.action_space.n
OBSERVATION_SPACE = env.observation_space.shape[0]
total_score = []
NUMOFFEATURES = 2

# Hiper parametros de la red
# Exploracion
EXPLORATION_MAX = 1
EXPLORATION_MIN = 0.3
# Parametros Q Learning
LEARNING_RATE = 0.0001
GAMMA = 0.99
HIDDEN_SIZE = 64
# Parametros Inverso
w = []
learner = None
expert = None
# Parametros de la memoria
MEMORY_SIZE = 10000
BATCH_SIZE = 200
PRETRAIN_LENGTH = BATCH_SIZE

# Creamos el objeto contenedor de la red neuronal y la memoria
tf.compat.v1.reset_default_graph()
QNetwork = NNAgent.QNetwork(name='QNetwork', hidden_size=HIDDEN_SIZE, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, state_size=OBSERVATION_SPACE, action_size=ACTION_SPACE)
memory = NNAgent.Memory(max_size=MEMORY_SIZE)

# Actualiza la red neuronal
def updateDeepQNetwork():
    global memory, QNetwork, sess
    # Sample mini-batch from memory
    batch = memory.sample(BATCH_SIZE)
    states = np.array([each[0] for each in batch])
    actions = np.array([each[1] for each in batch])
    rewards = np.array([each[2] for each in batch])
    next_states = np.array([each[3] for each in batch])
                
    # Train network
    target_Qs = sess.run(QNetwork.output, feed_dict={QNetwork.inputs_: next_states})
                
    # Set target_Qs to 0 for states where episode ends
    episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
    target_Qs[episode_ends] = (0, 0)
                
    targets = rewards + GAMMA * np.max(target_Qs, axis=1)

    loss, _ = sess.run([QNetwork.loss, QNetwork.opt],
        feed_dict={QNetwork.inputs_: states,
        QNetwork.targetQs_: targets,
        QNetwork.actions_: actions})

# Funcion Gausiana
def gaussian_function(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

# Obtiene las caracteristicas del estado (IRL)
def getFeatures(state):
    return [gaussian_function(state[0], 0, 0.5),gaussian_function(state[2], 0, 1)]

# Funcion para tomar una accion basado en el epsilon
def getAction(state,epsilon):
    global QNetwork,sess
    if np.random.rand() <= epsilon:
        # Toma una accion aleatoria
        action = random.randrange(0, ACTION_SPACE)
    else:
        # Toma una accion aprendida de la red neuronal
        feed = {QNetwork.inputs_: state.reshape((1, *state.shape))}
        Qs = sess.run(QNetwork.output, feed_dict=feed)
        action = np.argmax(Qs)
    return action

# Calculamos la utilidad en las demostraciones del aprendiz
def calc_feature_expectation(demonstrations):
    feature_expectations = np.zeros(NUMOFFEATURES)
    demo_num = len(demonstrations)
    
    for _ in range(demo_num):
        state = env.reset()
        demo_length = 0
        done = False
        steps=0
        while not done:
            steps+=1
            demo_length += 1
            action = getAction(state,0.0)
            next_state, _, done, _ = env.step(action)
            features = getFeatures(next_state)
            feature_expectations += (GAMMA**(demo_length)) * np.array(features)
            state = next_state
    feature_expectations = feature_expectations/ demo_num
    return feature_expectations

# Calculamos la utilidad en las demostraciones del experto
def expert_feature_expectation(demonstrations):
    feature_expectations = np.zeros(NUMOFFEATURES)
    for demo_num in range(len(demonstrations)):
        steps=0
        for demo_length in range(len(demonstrations[demo_num])):
            steps+=1
            state = demonstrations[demo_num][demo_length][0]
            features = getFeatures(state)
            feature_expectations += (GAMMA**(demo_length)) * np.array(features)
    feature_expectations = feature_expectations / len(demonstrations)
    return feature_expectations

def subtract_feature_expectation(learner):
    # if status is infeasible, subtract first feature expectation
    learner = learner[1:][:]
    return learner

def add_feature_expectation(learner, temp_learner):
    # save new feature expectation to list after RL step
    learner = np.vstack([learner, temp_learner])
    return learner

# Optimizador
def QP_optimizer(learner, expert):
    w = cp.Variable(NUMOFFEATURES)
    obj_func = cp.Minimize(cp.norm(w))
    constraints = [(expert-learner)*(w.T)>= 2] 
    prob = cp.Problem(obj_func, constraints)
    prob.solve()
    if prob.status == "optimal":
        print("status:", prob.status)
        print("optimal value", prob.value)
        weights = np.squeeze(np.asarray(w.value))
        return weights, prob.status
    else:
        print("status:", prob.status)
        weights = np.zeros(NUMOFFEATURES)
        return weights, prob.status

# Funcion para inicializar la memoria
def preTrain(samples):
    global memory
    # Reinicia el entorno
    state = env.reset()
    for _ in range(samples):
        # Toma una accion aleatoria
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        if done:
            # La simulacion falla porque no hay un estado siguiente
            next_state = np.zeros(state.shape)
            # Añade la experiencia a la memoria
            memory.add((state, action, reward, next_state))
            # Inicia un nuevo episodio
            env.reset()
            # Toma una accion aleatoria
            state, reward, done, _ = env.step(env.action_space.sample())
        else:
            # Añade la experiencia a la memoria y continua
            memory.add((state, action, reward, next_state))
            state = next_state

# Creamos la variable para la sesion actual
sess = tf.compat.v1.InteractiveSession()
# Inicializa las variables
sess.run(tf.compat.v1.global_variables_initializer())
# Almacena la sesion 
saver = tf.compat.v1.train.Saver()

# Funcion para cargar un modelo ya entrenado
def loadTrain(savedir = SAVEDIR):
    global w,total_score
    w = np.load(SAVEDIR+"w.npy",allow_pickle=True)
    total_score = np.load(SAVEDIR+"total_score.npy",allow_pickle=True)
    saver.restore(sess,savedir+'cartpole.ckpt')

# Funcion de entrenamiento
def train(episodes, expertPath = "./Cartpole/Expert.npy"):
    # Llama a las variables globales creadas anteriormente
    global QNetwork, saver, sess, total_score, w, learner, expert
    # Crea las demostraciones e inicializa la matriz del aprendiz
    demonstrations = np.load(expertPath,allow_pickle=True)
    learner = calc_feature_expectation(demonstrations)
    learner = np.matrix([learner])
    # Crea la matriz del experto
    expert = expert_feature_expectation(demonstrations)
    expert = np.matrix([expert])
    w, status = QP_optimizer(learner, expert)
    # Incializa la memoria
    preTrain(PRETRAIN_LENGTH)
    # Crea un arreglo con un epsilon lineal hasta la mitad de los episodios
    epsilon = list(np.linspace(EXPLORATION_MAX,EXPLORATION_MIN, round(episodes/2)))+list(np.ones(episodes-round(episodes/2))*EXPLORATION_MIN)
    total_score = []
    # Inicia el entrenamiento
    for episode in range(episodes):
        # Inicia un nuevo episodio
        state = env.reset()
        episodeReward = 0
        while True:
            action = getAction(state,epsilon[episode])  # Toma una accion
            next_state, reward, done, _ = env.step(action) # Ejecuta la accion en el entorno
            
            # IRL
            features = np.array(getFeatures(state))
            irl_reward = np.dot(w, features)

            episodeReward+=reward # Sumamos la recompensa a la recompensa total del episodio 
            next_state = np.zeros(state.shape) if done else next_state # Si termina el episodio, no hay un estado siguiente
            memory.add((state, action, irl_reward, next_state)) # Añade la iteracion a la memoria
            state = next_state # El siguiente estado se convierte en el actual
            updateDeepQNetwork() # Entrenamos la red neuronal con la memoria
            if done:
                total_score.append(episodeReward) # Añadimos la recompensa total del episodio al historico de recompensas
                break
        # Almacenamos el entrenamiento
        if episode % 10 == 0:
            # optimize weight per # episode
            status = "infeasible"
            temp_learner = calc_feature_expectation(demonstrations)
            learner = add_feature_expectation(learner, temp_learner)
            
            while status=="infeasible":
                w, status = QP_optimizer(learner, expert)
                if status=="infeasible":
                    learner = subtract_feature_expectation(learner)
            
            print("Episodio {0} recompensa = {1}, epsilon = {2}, w = {3}".format(episode, episodeReward, epsilon[episode],w))
            np.save(SAVEDIR+"w.npy",w)
            np.save(SAVEDIR+"total_score.npy",total_score)
            saver.save(sess, SAVEDIR+'cartpole.ckpt')
    print("Finalizo el entrenamiento!")

# Prueba el entrenamiento
def test(numtest, render = False):
    test_score = []
    for e in range(numtest):
        # Inicia un nuevo episodio
        state = env.reset()
        episodeReward = 0
        while True:
            if render:
                env.render()
            action = getAction(state,0.0)  # Toma una accion
            next_state, reward, done, _ = env.step(action) # Ejecuta la accion en el entorno
            episodeReward+=reward # Sumamos la recompensa a la recompensa total del episodio 
            state = next_state
            if done:
                test_score.append(episodeReward) # Añadimos la recompensa total del episodio al historico de recompensas
                break
        print("Prueba {0}, recompensa {1}".format(e,episodeReward))
    env.close()
    totalDone=sum([1 if i>490 else 0 for i in test_score])
    print("Num. Test = {0}, Completados = {1}, Porcentaje = {2}%".format(numtest,totalDone,(totalDone/numtest)*100))

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def printGraph(n=10,path=SAVEDIR, show = False):
    global total_score
    #total_score=np.load(path+'total_score.npy',allow_pickle=True)
    plt.clf()
    plt.plot(total_score)
    plt.plot(moving_average(total_score,n))
    plt.ylabel('Iteraciones hasta terminar')
    plt.xlabel('Episodios')
    plt.savefig(path+'total_score.png', dpi=600)
    if show:
        plt.show()