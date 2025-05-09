import retro
import pygame
import numpy as np
import time

# Inicializa o pygame
pygame.init()

# Cria o ambiente
env = retro.make(game='BubbleBobble-Nes')
obs = env.reset()

# Define o tamanho da janela
screen_width, screen_height = obs.shape[1] * 3, obs.shape[0] * 3
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Bubble Bobble - NES")

print("Controles:")
print("←/→/↑/↓ para movimentar | Z = A | X = B | ESC = sair")
print("Use o mouse para selecionar uma área e ver as cores únicas nessa região.")

action = [0] * 9

# Variáveis para controle da seleção
selecionando = False
inicio = None
fim = None

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Botão esquerdo do mouse
                selecionando = True
                inicio = pygame.mouse.get_pos()
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and selecionando:
                fim = pygame.mouse.get_pos()
                selecionando = False

                # Ordena os pontos para garantir coordenadas válidas
                x1, y1 = inicio
                x2, y2 = fim
                x_min, x_max = sorted([x1, x2])
                y_min, y_max = sorted([y1, y2])

                # Redimensiona para o tamanho original da tela do jogo
                x_min //= 3
                x_max //= 3
                y_min //= 3
                y_max //= 3

                # Garante que as coordenadas estão dentro da tela
                x_min = max(0, min(x_min, obs.shape[1] - 1))
                x_max = max(0, min(x_max, obs.shape[1]))
                y_min = max(0, min(y_min, obs.shape[0] - 1))
                y_max = max(0, min(y_max, obs.shape[0]))

                # Recorta e extrai as cores únicas
                recorte = obs[y_min:y_max, x_min:x_max]
                cores_unicas = np.unique(recorte.reshape(-1, 3), axis=0)
                print(f"\nCores únicas na região ({x_min}, {y_min}) até ({x_max}, {y_max}):")
                for cor in cores_unicas:
                    print(f"RGB: {cor}")

    # Verifica teclas pressionadas
    keys = pygame.key.get_pressed()
    action = [0] * 9
    if keys[pygame.K_z]: action[0] = 1
    if keys[pygame.K_x]: action[1] = 1
    if keys[pygame.K_w] or keys[pygame.K_UP]: action[8] = 1
    if keys[pygame.K_s] or keys[pygame.K_DOWN]: action[5] = 1
    if keys[pygame.K_a] or keys[pygame.K_LEFT]: action[7] = 1
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]: action[6] = 1

    # Executa a ação no ambiente
    obs, reward, done, info = env.step(action)

    # Rotaciona e converte para Surface
    obs_rgb = np.rot90(obs)
    obs_surface = pygame.surfarray.make_surface(obs_rgb)
    obs_surface = pygame.transform.scale(obs_surface, (screen_width, screen_height))

    # Desenha a tela
    screen.blit(obs_surface, (0, 0))

    # Se estiver selecionando, desenha o retângulo da seleção
    if selecionando and inicio:
        pos_atual = pygame.mouse.get_pos()
        rect = pygame.Rect(inicio, (pos_atual[0] - inicio[0], pos_atual[1] - inicio[1]))
        pygame.draw.rect(screen, (255, 255, 0), rect, 2)

    pygame.display.update()

    if done:
        obs = env.reset()

    time.sleep(1 / 60)

pygame.quit()
env.close()
