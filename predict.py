import cv2
import mediapipe as mp
import numpy as np

# Settings
w = 700
h = 400
# Opening camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
# Setting hands detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
# Draw points
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(138, 43, 226), thickness=7)  # Point style
handConStyle = mpDraw.DrawingSpec(color=(205, 16, 118), thickness=7)  # Line style


def main():
    # Creating array for all points that needs drawing
    drawPoints = []
    isDrawing = False
    # print(imgDraw)
    while True:
        ret, img = cap.read()
        if ret:
            # Get image and find the hand
            img = cv2.resize(img, (w, h))
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
            imgRGB = cv2.flip(imgRGB, 1)
            img = cv2.flip(img, 1)
            result = hands.process(imgRGB)
            # Setting the whole image to white
            img[:] = (255, 255, 255)
            # When there are hands in camera
            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    # Draw hands
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                    # Getting 21 points from mediapipe
                    hand_local = []
                    for i in range(21):  # Getting points
                        xPos = handLms.landmark[i].x * img.shape[1]
                        yPos = handLms.landmark[i].y * img.shape[0]
                        hand_local.append((xPos, yPos))
                    # If the dis of thumb tip and index tip less than 50, save the point at the moment
                    linearDistance = np.sqrt(np.square(hand_local[4][0]-hand_local[8][0])+np.square(hand_local[4][1]-hand_local[8][1]))
                    if linearDistance <= 30:
                        if isDrawing == False:
                            drawPoints.append([-1])

                        isDrawing = True
                        drawPoints.append([round(int(hand_local[4][0])), round(int(hand_local[4][1]))])
                        cv2.putText(img, str(round(hand_local[4][0]))+', '+str(round(hand_local[4][1])), (int(hand_local[4][0])-70, int(hand_local[4][1])+70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
                    else:
                        isDrawing = False
            # Draw lines between all points that is saved during lifecycle
            for p in range(len(drawPoints)-1):
                if drawPoints[p] != [-1]:
                    if drawPoints[p+1] == [-1]:
                        continue
                    cv2.line(img, drawPoints[p], drawPoints[p+1], (0, 0, 0), 5)

            # Text on the image
            cv2.rectangle(img, (200, 50), (500, 350), (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(img, 'Please draw inside the rectangle.', (5, 25), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 2)
            cv2.putText(img, 'Hands out when confirming the image.', (5, 385), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, 'E:Confirm', (5, 75), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img, 'W:Clean', (5, 120), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img, 'Q:Quit', (5, 165), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)

            # Show the image
            cv2.imshow('img', img)

        # Clear the image
        if cv2.waitKey(1) == ord('w'):
            drawPoints = []

        # Break condition
        if cv2.waitKey(1) == ord('q'):
            break

    # Releasing the camera
    cap.release(0)

main()

