from ast import Continue
import retro
import cv2
import numpy as np
import time
import random

def captura_tela(obs):
    return obs.copy()

# Função para detectar o personagem e retornar seu centro
def detectar_personagem(tela):
    hsv = cv2.cvtColor(tela, cv2.COLOR_RGB2HSV)

    cores_rgb = [
        [72, 220, 72],
        [248, 116, 96]
    ]

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

            # 👉 Desenhar retângulo ao redor do personagem
            x, y, w, h = cv2.boundingRect(maior_contorno)
            cv2.rectangle(tela, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Verde

    return contornos, mascara_total, centro_personagem

# NOVA FUNÇÃO: detecta inimigos (ex: tons de azul)
# Função para detectar inimigos e retornar seus centros
def detectar_inimigos(tela):
    hsv = cv2.cvtColor(tela, cv2.COLOR_RGB2HSV)

    # Cores RGB dos inimigos (convertidas para HSV)
    cores_rgb = [
        [0, 112, 232],
        [168, 228, 248],
        [248, 116, 176]
    ]

    # Converte RGB para HSV e define uma faixa com tolerância
    faixas = []
    for rgb in cores_rgb:
        hsv_val = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
        lower = np.clip(hsv_val - [10, 50, 50], 0, 255)
        upper = np.clip(hsv_val + [10, 50, 50], 0, 255)
        faixas.append((lower, upper))

    # Cria uma máscara combinada para todas as cores
    mascara_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in faixas:
        mascara = cv2.inRange(hsv, lower, upper)
        mascara_total = cv2.bitwise_or(mascara_total, mascara)

    # Encontra contornos dos inimigos detectados
    contornos, _ = cv2.findContours(mascara_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centros_inimigos = []  # Lista para armazenar os centros dos inimigos

    # Marcar o centro de cada inimigo (em amarelo) e armazenar as coordenadas do centro
    for contorno in contornos:
        M = cv2.moments(contorno)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(tela, (cx, cy), 1, (0, 255, 255), -1)  # Amarelo
            centros_inimigos.append((cx, cy))  # Armazenar o centro

    return contornos, mascara_total, centros_inimigos

# Função para repelir o personagem dos inimigos
# Função para repelir o personagem dos inimigos
def repulsao(personagem, inimigos, K, tela):
    movimento = np.array([0.0, 0.0])  # Inicializa o vetor de movimento

    if not inimigos:
        return movimento  # Se não houver inimigos, não há movimento

    # Calcula as distâncias entre o personagem e todos os inimigos
    distancias = []
    for centro_inimigo in inimigos:
        distancia = np.linalg.norm(personagem - centro_inimigo)  # Distância euclidiana
        distancias.append(distancia)

    # Encontra o índice do inimigo mais próximo
    indice_inimigo_mais_proximo = np.argmin(distancias)
    centro_inimigo_proximo = inimigos[indice_inimigo_mais_proximo]

    # Calcula o vetor de repulsão em relação ao inimigo mais próximo
    vetor_repulsao = personagem - centro_inimigo_proximo
    distancia_proxima = distancias[indice_inimigo_mais_proximo]

    # Evita movimento zero quando a distância é muito pequena
    if distancia_proxima < 50:  # Limite de distância para repulsão
        if distancia_proxima != 0:
            vetor_repulsao = vetor_repulsao / distancia_proxima  # Normaliza
        else:
            vetor_repulsao = np.array([1, 0])  # Movimento arbitrário se a distância for zero

        movimento += K * vetor_repulsao

        # Desenha a seta de repulsão para visualização
        cv2.arrowedLine(
            tela,
            tuple(personagem.astype(int)),
            tuple((personagem + vetor_repulsao * 10).astype(int)),
            (255, 255, 255),
            2,
            tipLength=0.4
        )

    return movimento

# Criação do ambiente
# Criação do ambiente
env = retro.make(game='BubbleBobble-Nes')
obs = env.reset()

altura, largura, _ = obs.shape
roi_x1, roi_y1 = 6, 25
roi_x2, roi_y2 = 233, 209

personagem = None  # Inicializa o personagem
contador_duracao = 0
tempo_maximo = 30

# Variável global para controlar o tempo de movimento aleatório
escolha = 0
tempo_aleatorio = 0  # Em segundos
limite_aleatorio = 1  # Tempo máximo de movimento aleatório (em segundos)

# Função para determinar a ação com base no movimento e no tempo de movimento aleatório
def acao_com_base_na_repulsao(movimento):
    global tempo_aleatorio  # Usando a variável global para controlar o tempo de movimento aleatório
    global escolha  # Usando a variável global para controlar o tempo de movimento aleatório
    acao = [0] * 9

    print(f"Movimento calculado: {movimento}")

    if np.allclose(movimento, [0.0, 0.0]):
        # Se o movimento for nulo (0,0), começa a contar o tempo para o movimento aleatório
        if tempo_aleatorio == 0:
            print("teste")
            tempo_aleatorio = time.time()  # Marca o início do tempo
            escolha = random.choice([6, 7])  # Escolhe aleatoriamente entre esquerda (6) ou direita (7)

        # Verifica se o tempo de movimento aleatório expirou
        if time.time() - tempo_aleatorio < limite_aleatorio:
            # Movimento aleatório ativo
            acao[escolha] = 1
            pass
        else:
            # Se o tempo de movimento aleatório expirar ou o movimento parar de ser 0,0, o movimento aleatório é desativado
            tempo_aleatorio = 0  # Marca o início do tempo
            escolha = random.choice([6, 7])  # Escolhe aleatoriamente entre esquerda (6) ou direita (7)
            acao[escolha] = 1
    else:
        # Se o movimento não for nulo, desativa o movimento aleatório e move de acordo com a repulsão
        tempo_aleatorio = 0  # Resetando o contador do movimento aleatório
        if movimento[0] < 0:
            acao[6] = 1  # Esquerda
        elif movimento[0] > 0:
            acao[7] = 1  # Direita
        if movimento[1] > 0:
            acao[8] = 1  # Pulo

    print(f"Ação gerada: {acao}")
    return acao

while True:
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == 27:
        break

    tela = captura_tela(obs)
    cv2.rectangle(tela, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)
    roi = tela[roi_y1:roi_y2, roi_x1:roi_x2]

    # Detecta personagem (verde)
    contornos_verde, _, centro_personagem = detectar_personagem(roi)

    # Se o personagem foi detectado, ajusta sua posição
    if centro_personagem is not None:
        centro_x, centro_y = centro_personagem

        centro_x_corr = centro_x + roi_x1
        centro_y_corr = centro_y + roi_y1

        #cv2.circle(tela, (centro_x_corr, centro_y_corr), 1, (0, 0, 255), -1)  # Desenha o centro do personagem

        # Imprime as coordenadas do personagem
        print(f"Coordenadas do personagem: ({centro_x_corr}, {centro_y_corr})")
        personagem = np.array([centro_x_corr, centro_y_corr])  # Atualiza a posição do personagem

    # Detecta inimigos (azul)
    contornos_inimigos, _, centros_inimigos = detectar_inimigos(roi)
    inimigos = []
    for contorno, centro in zip(contornos_inimigos, centros_inimigos):
        x, y, w, h = cv2.boundingRect(contorno)
        x_corr = x + roi_x1
        y_corr = y + roi_y1
        cv2.rectangle(tela, (x_corr, y_corr), (x_corr + w, y_corr + h), (0, 0, 255), 2)

        # Armazenar os centros corrigidos (com a posição do ROI)
        centro_corrigido = (centro[0] + roi_x1, centro[1] + roi_y1)
        inimigos.append(centro_corrigido)

    # Repulsão do personagem pelos inimigos
    if personagem is not None:
        movimento = repulsao(personagem, inimigos, 1.0, tela)

        # Realiza a ação com base no movimento calculado pela repulsão
        action = acao_com_base_na_repulsao(movimento)
    else:
        # Caso o personagem não tenha sido detectado, mover aleatoriamente
        action = [random.randint(0, 1) for _ in range(9)]  # Gera uma ação aleatória de 0 ou 1 para cada uma das 9 ações possíveis

    # Exibe a tela com as atualizações
    tela_maior = cv2.resize(tela, (largura * 3, altura * 3), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Tela do jogo", tela_maior)

    # Executa a ação no ambiente e obtém o próximo estado
    print(f"Ação geradak: {action}")
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

    print(f"Ação gerada1k: {action}")
    # Atraso para ajustar a taxa de quadros
    time.sleep(1 / 60)

cv2.destroyAllWindows()
env.close()
