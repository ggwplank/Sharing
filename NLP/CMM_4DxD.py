import numpy as np

### Generate the vocabulary
def generate_vocabulary(text):
    
    return list(dict.fromkeys(text.lower().split()))


### We retrieve ALL the words (even the replicated) from our text.txt
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

### We define our vocabulary, composed of unique terms in the text.txt
vocabulary = generate_vocabulary(text)
print("Vocabulary size:", len(vocabulary))

words = generate_words(text)

### We define the window_size, that is the length of the sequence we want to consider in order to predict the newt word
window_size = 4
print("Window size:", window_size, "\n")

### We create one-hot vectors for each word in our vocabulary (so its size is based on our vocabulary's size)
onehot_vectors = np.identity(len(vocabulary))
print(f"There are {len(onehot_vectors)} one-hot vectors for each word in the vocabulary:\n", onehot_vectors, "\n")


### This return the index of a specific word in our pre-defined vocabulary
def index(word):
    return vocabulary.index(word)


print(f"We combine our one-hot vectors from {window_size}x{len(vocabulary)} to 1x{window_size * len(vocabulary)}\n")
### We are combining our one-hot vectors into a single one.
def seq(vector):
    s1 = vector[0]
    s2 = vector[1]
    s3 = vector[2]
    s4 = vector[3]

    combined_vector = onehot_vectors[[index(s1), index(s2), index(s3), index(s4)]].reshape(1, window_size * len(vocabulary))
    return combined_vector


def create_CMM():
    CMM = np.zeros((window_size * len(vocabulary), len(vocabulary)))

    for s in range(len(words) - window_size):
        sequence = words[s:s + window_size]
        target = words[s + window_size]

        m1 = np.transpose(seq(sequence))
        m2 = onehot_vectors[[index(target)]]

        CMM += np.matmul(m1, m2)

    print("Combined one-hot vector transposed x one-hot vector of the target")
    print(np.shape(m1), "x", np.shape(m2))

    return CMM


CMM = create_CMM()
print("\nCorrelation Matrix (CMM):", np.shape(CMM), "\n", CMM, "\n")

current_text = "La tecnologia ha trasformato"
print(current_text)

for i in range(window_size, len(words)):
    vector = words[i - 4:i]
    output = np.matmul(seq(vector),CMM)
    print(vocabulary[np.argmax(output)])

