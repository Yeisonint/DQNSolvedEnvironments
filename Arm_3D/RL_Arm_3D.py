# Librerias
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

# Cartpole
import arm3d
import NNAgent

# Parametros del entorno
SAVEDIR = './Arm_3D/Train_RL/'
env = arm3d.ArmEnv3D()
ACTION_SPACE = len(env._ACTIONS)
OBSERVATION_SPACE = env._STATE_DIM
total_score = []

# Hiper parametros de la red
# Exploracion
EXPLORATION_MAX = 1
EXPLORATION_MIN = 0.3
# Parametros Q Learning
LEARNING_RATE = 0.0001
GAMMA = 0.99
HIDDEN_SIZE = 64
# Parametros de la memoria
MEMORY_SIZE = 10000
BATCH_SIZE = 200
PRETRAIN_LENGTH = BATCH_SIZE

# Creamos el objeto contenedor de la red neuronal y la memoria
tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()
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

# Funcion para tomar una accion basado en el epsilon
def getAction(state,epsilon):
    global QNetwork,sess
    if np.random.rand() <= epsilon:
        # Toma una accion aleatoria
        action = random.randrange(0, len(ACTION_SPACE)**env._ACTION_DIM)
    else:
        # Toma una accion aprendida de la red neuronal
        feed = {QNetwork.inputs_: state.reshape((1, *state.shape))}
        Qs = sess.run(QNetwork.output, feed_dict=feed)
        action = np.argmax(Qs)
    return action

# Funcion para inicializar la memoria
def preTrain(samples):
    global memory
    # Reinicia el entorno
    state = env.reset()
    for _ in range(samples):
        # Toma una accion aleatoria
        action = random.randrange(0, len(ACTION_SPACE)**env._ACTION_DIM)
        next_state, reward, done, _ = env.step(action)
        if done:
            # La simulacion falla porque no hay un estado siguiente
            next_state = np.zeros(state.shape)
            # Añade la experiencia a la memoria
            memory.add((state, action, reward, next_state))
            # Inicia un nuevo episodio
            env.reset()
            # Toma una accion aleatoria
            state, reward, done, _ = env.step(random.randrange(0, len(ACTION_SPACE)**env._ACTION_DIM))
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
    global total_score
    total_score = np.load(SAVEDIR+"total_score.npy",allow_pickle=True)
    saver.restore(sess,savedir+'cartpole.ckpt')

# Funcion de entrenamiento
def train(episodes,rewardType=0):
    # Llama a las variables globales creadas anteriormente
    global QNetwork, saver, sess, total_score
    # Incializa la memoria
    preTrain(PRETRAIN_LENGTH)
    # Crea un arreglo con un epsilon lineal hasta la mitad de los episodios
    epsilon = list(np.linspace(EXPLORATION_MAX,EXPLORATION_MIN, round(episodes/2)))+list(np.ones(episodes-round(episodes/2))*EXPLORATION_MIN)
    total_score = []
    # Inicia el entrenamiento
    for episode in range(1,episodes+1):
        # Inicia un nuevo episodio
        state = env.reset()
        episodeReward = 0
        while True:
            action = getAction(state,epsilon[episode-1])  # Toma una accion
            next_state, reward, done, _ = env.step(action) # Ejecuta la accion en el entorno
            #reward=rewardFunc(state,rew,rewardType) # Creamos una funcion propia para las recompensas
            episodeReward+=reward # Sumamos la recompensa a la recompensa total del episodio 
            next_state = np.zeros(state.shape) if done else next_state # Si termina el episodio, no hay un estado siguiente
            memory.add((state, action, reward, next_state)) # Añade la iteracion a la memoria
            state = next_state # El siguiente estado se convierte en el actual
            updateDeepQNetwork() # Entrenamos la red neuronal con la memoria
            if done:
                total_score.append(episodeReward) # Añadimos la recompensa total del episodio al historico de recompensas
                break
        # Almacenamos el entrenamiento
        if episode % 10 == 0:
            print("Episodio {0} recompensa = {1}, epsilon = {2}".format(episode, episodeReward, epsilon[episode-1]))
            np.save(SAVEDIR+"total_score.npy",total_score)
            saver.save(sess, SAVEDIR+'cartpole.ckpt')
    print("Finalizo el entrenamiento!")

# Prueba el entrenamiento
def test(numtest, render = False):
    test_score = []
    for e in range(1,numtest+1):
        # Inicia un nuevo episodio
        print("Prueba {0}".format(e))
        state = env.reset()
        episodeReward = 0
        while True:
            if render:
                env.render()
            action = getAction(state,0.0)  # Toma una accion
            next_state, rew, done, _ = env.step(action) # Ejecuta la accion en el entorno
            reward=rewardFunc(state,rew,0) # Creamos una funcion propia para las recompensas
            episodeReward+=reward # Sumamos la recompensa a la recompensa total del episodio 
            state = next_state
            if done:
                test_score.append(episodeReward) # Añadimos la recompensa total del episodio al historico de recompensas
                break
    env.close()
    totalDone=sum([1 if i>490 else 0 for i in test_score])
    print("Num. Test = {0}, Completados = {1}, Porcentaje = {2}%".format(numtest,totalDone,(totalDone/numtest)*100))

def gaussian_function(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# Creamos las trayectorias desde el experto
def makeExpertPath(numofdemo,safedir="./Cartpole/"):
    demos = []
    for e in range(1,numofdemo+1):
        state = env.reset()
        demo=[]
        steps = 0
        while True:
            action = getAction(state,0.0)  # Toma una accion
            next_state, rew, done, _ = env.step(action) # Ejecuta la accion en el entorno
            reward=rewardFunc(state,rew,1) # Creamos una funcion propia para las recompensas
            demo.append(np.array([state,action,reward,next_state]))
            state = next_state
            steps+=1
            if done:
                break
        demos.append(demo)
        print("Num =  {0}, Steps = {1}".format(e,steps))
    np.save(safedir+"Expert.npy",demos)

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
