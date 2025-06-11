from ast import Continue
import retro
import cv2
import numpy as np
import time
import random

# Classe para representar um inimigo com filtro de Kalman
class KalmanInimigo:
    def __init__(self, pos_inicial):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.statePre = np.array([[np.float32(pos_inicial[0])],
                                         [np.float32(pos_inicial[1])],
                                         [0],
                                         [0]])
        self.ultima_posicao = pos_inicial
        self.frames_sem_atualizar = 0

    def atualizar(self, medida):
        medida_np = np.array([[np.float32(medida[0])],
                              [np.float32(medida[1])]])
        self.kalman.correct(medida_np)
        self.ultima_posicao = medida
        self.frames_sem_atualizar = 0

    def prever(self):
        pred = self.kalman.predict()
        return (int(pred[0]), int(pred[1]))

def captura_tela(obs):
    return obs.copy()

def detectar_personagem(tela):
    hsv = cv2.cvtColor(tela, cv2.COLOR_RGB2HSV)
    cores_rgb = [[72, 220, 72], [248, 116, 96]]
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
    centro_personagem = None

    if contornos:
        maior_contorno = max(contornos, key=cv2.contourArea)
        M = cv2.moments(maior_contorno)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centro_personagem = (cx, cy)
            cv2.circle(tela, (cx, cy), 1, (0, 255, 255), -1)
            x, y, w, h = cv2.boundingRect(maior_contorno)
            cv2.rectangle(tela, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return contornos, mascara_total, centro_personagem

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

def repulsao(personagem, inimigos, K, tela):
    movimento = np.array([0.0, 0.0])
    if not inimigos:
        return movimento

    distancias = [np.linalg.norm(personagem - np.array(i)) for i in inimigos]
    indice = np.argmin(distancias)
    inimigo_proximo = np.array(inimigos[indice])
    vetor = personagem - inimigo_proximo
    dist = distancias[indice]

    if dist < 50:
        if dist != 0:
            vetor = vetor / dist
        else:
            vetor = np.array([1, 0])
        movimento += K * vetor
        cv2.arrowedLine(tela, tuple(personagem.astype(int)),
                        tuple((personagem + vetor * 10).astype(int)),
                        (255, 255, 255), 2, tipLength=0.4)

    return movimento

env = retro.make(game='BubbleBobble-Nes')
obs = env.reset()
altura, largura, _ = obs.shape
roi_x1, roi_y1 = 6, 25
roi_x2, roi_y2 = 233, 209

personagem = None
escolha = 0
tempo_aleatorio = 0
limite_aleatorio = 1

rastreadores = []

def acao_com_base_na_repulsao(movimento):
    global tempo_aleatorio, escolha
    acao = [0] * 9

    if np.allclose(movimento, [0.0, 0.0]):
        if tempo_aleatorio == 0:
            tempo_aleatorio = time.time()
            escolha = random.choice([6, 7])
        if time.time() - tempo_aleatorio < limite_aleatorio:
            acao[escolha] = 1
        else:
            tempo_aleatorio = 0
    else:
        tempo_aleatorio = 0
        if movimento[0] < 0:
            acao[6] = 1
        elif movimento[0] > 0:
            acao[7] = 1
        if movimento[1] > 0:
            acao[8] = 1

    return acao

while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

    tela = captura_tela(obs)
    cv2.rectangle(tela, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)
    roi = tela[roi_y1:roi_y2, roi_x1:roi_x2]

    _, _, centro_personagem = detectar_personagem(roi)
    if centro_personagem:
        personagem = np.array([centro_personagem[0] + roi_x1, centro_personagem[1] + roi_y1])

    _, _, centros_detectados = detectar_inimigos(roi)

    # Atualização dos rastreadores
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
            novos_rastreadores.append(KalmanInimigo(centro_corrigido))

    rastreadores = novos_rastreadores

    inimigos = [r.prever() for r in rastreadores]
    for i in inimigos:
        cv2.circle(tela, i, 3, (255, 0, 255), -1)

    if personagem is not None:
        mov = repulsao(personagem, inimigos, 1.0, tela)
        action = acao_com_base_na_repulsao(mov)
    else:
        action = [random.randint(0, 1) for _ in range(9)]

    tela_maior = cv2.resize(tela, (largura * 3, altura * 3), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Tela do jogo", tela_maior)

    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

    time.sleep(1 / 60)

cv2.destroyAllWindows()
env.close()
