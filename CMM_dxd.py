import numpy as np


def generate_vocabulary(text):
    # Rimuovi eventuali segni di punteggiatura e rendi tutte le parole minuscole
    VOC = list(dict.fromkeys(text.lower().split()))  # Rimuove le ripetizioni mantenendo l'ordine
    return VOC


def generate_words(text):
    return text.lower().split()


text = ("La tecnologia ha trasformato radicalmente il modo in cui viviamo e lavoriamo, influenzando quasi ogni aspetto "
        "della nostra vita quotidiana. Negli ultimi decenni, l'avvento di Internet ha rivoluzionato la comunicazione, "
        "rendendo possibile connettersi con persone in tutto il mondo con pochi clic. Inoltre, l'evoluzione degli "
        "smartphone ha portato a una disponibilità di informazioni praticamente illimitata a portata di mano. Oggi, "
        "possiamo accedere a notizie, social media, video, e applicazioni che ci permettono di lavorare, imparare, "
        "e divertirci, tutto tramite un dispositivo tascabile. Anche il settore della salute ha subito grandi "
        "cambiamenti grazie alla tecnologia. L'intelligenza artificiale viene utilizzata per analizzare enormi "
        "quantità di dati medici, aiutando i medici a fare diagnosi più accurate e a sviluppare piani di trattamento "
        "personalizzati. Inoltre, dispositivi come smartwatch e fitness tracker monitorano la nostra attività fisica, "
        "il sonno, e il battito cardiaco, permettendoci di prendere decisioni informate sulla nostra salute. La "
        "telemedicina, che consente ai pazienti di consultare i medici da remoto, è diventata sempre più popolare, "
        "soprattutto durante la pandemia di COVID-19, rendendo le cure accessibili a un numero maggiore di persone. "
        "Nel mondo del lavoro, la digitalizzazione ha cambiato la natura di molte professioni. Mentre alcune attività "
        "ripetitive sono state automatizzate, sono emerse nuove opportunità nel campo della programmazione, "
        "della gestione dei dati e della cybersecurity. Molte aziende ora permettono ai loro dipendenti di lavorare "
        "da casa, grazie alle piattaforme di collaborazione online e agli strumenti di videoconferenza. Questo ha "
        "portato a una maggiore flessibilità per i lavoratori, ma anche a nuove sfide, come il mantenimento di un "
        "equilibrio tra vita privata e lavoro. La tecnologia ha anche avuto un impatto significativo sull'istruzione. "
        "Le piattaforme di apprendimento online offrono corsi in una vasta gamma di argomenti, accessibili a chiunque "
        "abbia una connessione Internet. Questo ha reso possibile per molte persone acquisire nuove competenze e "
        "migliorare la loro formazione, indipendentemente dalla loro posizione geografica o disponibilità economica. "
        "Nonostante i numerosi vantaggi, la tecnologia presenta anche sfide e rischi. La dipendenza dai dispositivi "
        "digitali può influire negativamente sulla nostra salute mentale, e la protezione dei dati personali è "
        "diventata una questione cruciale. Inoltre, l'automazione e l'intelligenza artificiale sollevano "
        "preoccupazioni riguardo alla perdita di posti di lavoro in determinati settori. In definitiva, la tecnologia "
        "è un potente strumento che ha il potenziale di migliorare la nostra vita in molti modi, ma deve essere "
        "gestita con attenzione per garantire che i benefici siano equamente distribuiti ciao")

VOC = generate_vocabulary(text)
words = generate_words(text)
onehot_vectors = np.identity(len(VOC))
len_seq = 4


def index(word):
    return VOC.index(word)


def seq(vector):
    s1 = vector[0]
    s2 = vector[1]
    s3 = vector[2]
    s4 = vector[3]

    return onehot_vectors[[index(s1), index(s2), index(s3), index(s4)]].reshape(1, len_seq * len(VOC))


def create_CMM(RM):
    temp = np.zeros((len(VOC), len(VOC)))

    for s in range(len(words) - 4):
        sequence = words[s:s + 4]
        target = words[s + 4]

        v1 = np.transpose(seq(sequence))
        #v1 è un vettore di 992 mentre RM 248x992 quindi il risultato è una colonna di 248
        v1_reduced = np.matmul(RM, v1)

        v2 = onehot_vectors[[index(target)]]

        temp += np.matmul(v1_reduced, v2)

    return temp


def generate_random_matrix(rows, cols, mean=0, std_dev=1):
    return np.random.normal(loc=mean, scale=std_dev, size=(rows, cols))



RM = generate_random_matrix(len(VOC), len_seq * len(VOC))
CMM = create_CMM(RM)

predicted_text = "La tecnologia ha trasformato"

for i in range(4, len(words)):
    vector = words[i - 4:i]

    sequence = seq(vector)

    reduced_vector = np.matmul(sequence, np.transpose(RM))

    out = np.matmul(reduced_vector, CMM)

    predicted_text = predicted_text + " " + VOC[np.argmax(out)]

print(predicted_text)

