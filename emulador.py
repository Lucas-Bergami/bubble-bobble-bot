import retro
import cv2
import numpy as np
import time
import random
import pickle
from collections import defaultdict

# ===============================
# FUNÇÕES DE KALMAN E DETECÇÃO (sem mudanças)
# ===============================
# ... (mantém sua classe KalmanGeral, funções detectar_personagem, observar_personagem etc)

class KalmanGeral:
    def __init__(self, pos_inicial):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        self.kalman.statePost = np.array([[np.float32(pos_inicial[0])], [np.float32(pos_inicial[1])], [0], [0]], dtype=np.float32)
        self.ultima_posicao = pos_inicial
        self.frames_sem_atualizar = 0

    def atualizar(self, medida):
        medida_np = np.array([[np.float32(medida[0])], [np.float32(medida[1])]])
        self.kalman.correct(medida_np)
        self.ultima_posicao = medida
        self.frames_sem_atualizar = 0

    def prever(self):
        pred = self.kalman.predict()
        self.frames_sem_atualizar += 1
        return (int(pred[0]), int(pred[1]))

    def get_posicao_estimada(self):
        estado = self.kalman.statePost
        return (int(estado[0]), int(estado[1]))

def captura_tela(obs):
    return obs.copy()

def detectar_personagem(tela):
    hsv = cv2.cvtColor(tela, cv2.COLOR_RGB2HSV)
    cores_rgb = [[248, 116, 96], [72, 220, 72], [96, 116, 248]]
    mascaras_individuais = []
    for rgb in cores_rgb:
        hsv_val = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        lower = np.clip(hsv_val - [10, 60, 60], 0, 255)
        upper = np.clip(hsv_val + [10, 60, 60], 0, 255)
        mascara = cv2.inRange(hsv, lower, upper)
        mascaras_individuais.append(mascara)
    mascara_total = np.zeros_like(mascaras_individuais[0])
    for m in mascaras_individuais:
        mascara_total = cv2.bitwise_or(mascara_total, m)
    contornos, _ = cv2.findContours(mascara_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 10:
            x, y, w, h = cv2.boundingRect(c)
            cores_presentes = sum(cv2.countNonZero(m[y:y+h, x:x+w]) > 5 for m in mascaras_individuais)
            if cores_presentes >= 2:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
    return None

kalman_personagem = None

def observar_personagem(tela):
    global kalman_personagem
    posicao = detectar_personagem(tela)
    if kalman_personagem is None:
        if posicao is not None:
            kalman_personagem = KalmanGeral(posicao)
            return posicao
        else:
            return None
    if posicao is not None:
        kalman_personagem.atualizar(posicao)
    return kalman_personagem.prever()

def detectar_inimigos(tela):
    hsv = cv2.cvtColor(tela, cv2.COLOR_RGB2HSV)
    cores_rgb = [[0, 112, 232], [168, 228, 248], [248, 116, 176]]
    faixas = []
    for rgb in cores_rgb:
        hsv_val = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        lower = np.clip(hsv_val - [10, 50, 50], 0, 255)
        upper = np.clip(hsv_val + [10, 50, 50], 0, 255)
        faixas.append((lower, upper))
    mascara_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in faixas:
        mascara = cv2.inRange(hsv, lower, upper)
        mascara_total = cv2.bitwise_or(mascara_total, mascara)
    contornos, _ = cv2.findContours(mascara_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centros_inimigos = []
    for contorno in contornos:
        M = cv2.moments(contorno)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centros_inimigos.append((cx, cy))
            cv2.circle(tela, (cx, cy), 1, (0, 255, 255), -1)
    return contornos, mascara_total, centros_inimigos

# ===============================
# FUNÇÕES DE Q-LEARNING
# ===============================
acoes = ['idle', 'esquerda', 'direita', 'pular', 'atirar']
Q = defaultdict(lambda: {a: 0.0 for a in acoes})
alpha = 0.3
gamma = 0.8
epsilon = 0.2
recompensas_ep = []

try:
    with open("q_table.pkl", "rb") as f:
        Q.update(pickle.load(f))
        print("Q-table carregada com sucesso.")
except FileNotFoundError:
    print("Q-table não encontrada. Treinamento do zero.")

def quantizar_posicao(pos, eixox=10, eixoy=8):
    x, y = pos
    qx = min(x // (320 // eixox), eixox - 1)
    qy = min(y // (240 // eixoy), eixoy - 1)
    return int(qx), int(qy)

def obter_estado(personagem, inimigos):
    if personagem is None or not inimigos:
        return ("desconhecido",)
    px, py = quantizar_posicao(personagem)
    inimigo_mais_proximo = min(inimigos, key=lambda i: np.linalg.norm(np.array(personagem) - np.array(i)))
    ix, iy = quantizar_posicao(inimigo_mais_proximo)
    return (px, py, ix, iy)

def escolher_acao(estado, modo_treinamento=True):
    if modo_treinamento and random.random() < epsilon:
        return random.choice(acoes)
    return max(Q[estado], key=Q[estado].get)

def atualizar_q(estado, acao, recompensa, proximo_estado):
    melhor_q_proximo = max(Q[proximo_estado].values()) if proximo_estado in Q else 0
    Q[estado][acao] += alpha * (recompensa + gamma * melhor_q_proximo - Q[estado][acao])

def codificar_acao(acao_nome):
    action = [0] * 9
    action[0] = 1
    if acao_nome == 'esquerda':
        action = [0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif acao_nome == 'direita':
        action = [0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif acao_nome == 'pular':
        action = [0, 0, 0, 0, 0, 0, 0, 0, 1]
    elif acao_nome == 'atirar':
        action = [0, 0, 0, 0, 0, 1, 0, 0, 0]
    return action

# ===============================
# LOOP PRINCIPAL DO JOGO
# ===============================

env = retro.make(game='BubbleBobble-Nes')
n_episodios_treinamento = 500
episodio = 0
modo_treinamento = True
obs = env.reset()
altura, largura, _ = obs.shape
roi_x1, roi_y1 = 6, 25
roi_x2, roi_y2 = 233, 209
kalman_personagem = None
rastreadores = []
recompensa_total = 0

while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

    tela = obs.copy()
    roi = tela[roi_y1:roi_y2, roi_x1:roi_x2]
    centro_estimado = observar_personagem(roi)
    personagem = np.array([centro_estimado[0] + roi_x1, centro_estimado[1] + roi_y1]) if centro_estimado else None
    _, _, centros_detectados = detectar_inimigos(roi)
    novos_rastreadores = []
    for centro in centros_detectados:
        centro_corrigido = (centro[0] + roi_x1, centro[1] + roi_y1)
        rastreador_encontrado = False
        for rastreador in rastreadores:
            if np.linalg.norm(np.array(rastreador.ultima_posicao) - np.array(centro_corrigido)) < 25:
                rastreador.atualizar(centro_corrigido)
                novos_rastreadores.append(rastreador)
                rastreador_encontrado = True
                break
        if not rastreador_encontrado:
            novos_rastreadores.append(KalmanGeral(centro_corrigido))
    rastreadores = novos_rastreadores
    inimigos = [r.prever() for r in rastreadores]
    estado_atual = obter_estado(personagem, inimigos)
    acao_nome = escolher_acao(estado_atual, modo_treinamento)
    action = codificar_acao(acao_nome)
    obs, reward, done, info = env.step(action)

    nova_tela = obs.copy()
    nova_roi = nova_tela[roi_y1:roi_y2, roi_x1:roi_x2]
    novo_personagem = observar_personagem(nova_roi)
    _, _, novos_inimigos = detectar_inimigos(nova_roi)
    novo_estado = obter_estado(novo_personagem, novos_inimigos)
    
    num_inimigos_antes = len(inimigos)

    obs, reward, done, info = env.step(action)

    nova_tela = obs.copy()
    nova_roi = nova_tela[roi_y1:roi_y2, roi_x1:roi_x2]
    novo_personagem = observar_personagem(nova_roi)
    _, _, novos_inimigos = detectar_inimigos(nova_roi)
    novo_estado = obter_estado(novo_personagem, novos_inimigos)
    num_inimigos_depois = len(novos_inimigos)

    recompensa = -1  # Penalidade base

    if acao_nome == 'atirar':
        if num_inimigos_depois < num_inimigos_antes:
            recompensa = +20  # Atirou e eliminou inimigo
        else:
            recompensa = -5   # Atirou à toa
    elif num_inimigos_depois == 0:  # Sem inimigos na tela
        alvo = np.array([120 + roi_x1, 40 + roi_y1])  # Ponto central no topo
        if personagem is not None:
            dist_antes = np.linalg.norm(np.array(personagem) - alvo)
        else:
            dist_antes = 9999
        if novo_personagem is not None:
            dist_depois = np.linalg.norm(np.array(novo_personagem) - alvo)
        else:
            dist_depois = dist_antes
        if dist_depois < dist_antes:
            recompensa = +1  # Aproximou-se do topo
        else:
            recompensa = -1  # Afastou-se do topo
    else:  # Inimigos ainda presentes
        if novo_personagem is not None and novos_inimigos:
            dist = np.linalg.norm(np.array(novo_personagem) - np.array(novos_inimigos[0]))
            if dist < 2:
                recompensa = -100  # Colado no inimigo, punição máxima
            else:
                recompensa = 0    # Distância segura, sem recompensa
        else:
            recompensa = -1  # Não detectado, penalidade genérica


    if done:
        recompensa = -1000


    recompensa_total += recompensa
    if modo_treinamento:
        atualizar_q(estado_atual, acao_nome, recompensa, novo_estado)

    if done:
        episodio += 1
        recompensas_ep.append(recompensa_total)
        print(f"Episódio {episodio} - Recompensa total: {recompensa_total}")
        recompensa_total = 0
        obs = env.reset()
        kalman_personagem = None
        rastreadores = []
        if episodio >= n_episodios_treinamento:
            modo_treinamento = False
            epsilon = 0
            with open("q_table.pkl", "wb") as f:
                pickle.dump(dict(Q), f)
            print("Fim do treinamento. Modo execução iniciado.")
            with open("recompensas.csv", "w") as f:
                f.write("episodio,recompensa\n")
                for i, r in enumerate(recompensas_ep):
                    f.write(f"{i+1},{r}\n")

    tela_maior = cv2.resize(tela, (largura * 3, altura * 3), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Tela do jogo", tela_maior)
    time.sleep(1 / 60)

cv2.destroyAllWindows()
env.close()
