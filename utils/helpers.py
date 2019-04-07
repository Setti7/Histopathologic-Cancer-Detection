import cv2


def balance_data(df):
    df = df.reset_index()

    positives = df.loc[df['label'] == 1]
    negatives = df.loc[df['label'] == 0]

    smallest_data = min(len(positives), len(negatives))

    positives = positives[:smallest_data]
    negatives = negatives[:smallest_data]

    df = positives.append(negatives).sort_values('index')

    return df


def show_img(img, label='image'):
    cv2.imshow(label, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
