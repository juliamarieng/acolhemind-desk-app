import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


MODEL_NAME = "vosk-model-small-pt-0.3"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
SAMPLE_RATE = 16000
MAX_RECORDING_SECONDS = 300
MAX_MODEL_DOWNLOAD_BYTES = 100 * 1024 * 1024
MAX_MODEL_EXTRACTED_BYTES = 200 * 1024 * 1024


class AudioServiceError(Exception):
    """Falha recuperável ao gravar ou transcrever áudio localmente."""


def _modelo_valido(caminho):
    """Aceita os dois formatos de modelo do Vosk.

    Formato novo:   pastas "am" e "conf" dentro do modelo.
    Formato antigo: arquivos soltos na raiz (final.mdl, mfcc.conf, Gr.fst...)
                    mais a pasta "ivector". É o caso do vosk-model-small-pt-0.3.
    """
    formato_novo = (caminho / "am").is_dir() and (caminho / "conf").is_dir()
    formato_antigo = (caminho / "final.mdl").is_file() and (
        caminho / "ivector"
    ).is_dir()
    return formato_novo or formato_antigo


def _diretorio_modelos():
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        base = str(Path.home())
    return Path(base) / "AcolheMind" / "models"


def _baixar_modelo(arquivo, progresso):
    requisicao = urllib.request.Request(
        MODEL_URL, headers={"User-Agent": "AcolheMind/1.0"}
    )
    tamanho_total = 0

    with urllib.request.urlopen(requisicao, timeout=30) as resposta:
        tamanho_informado = int(resposta.headers.get("Content-Length", "0"))
        if tamanho_informado > MAX_MODEL_DOWNLOAD_BYTES:
            raise AudioServiceError("O arquivo do modelo excede o limite permitido.")

        with arquivo.open("wb") as destino:
            while True:
                bloco = resposta.read(64 * 1024)
                if not bloco:
                    break
                tamanho_total += len(bloco)
                if tamanho_total > MAX_MODEL_DOWNLOAD_BYTES:
                    raise AudioServiceError(
                        "O arquivo do modelo excede o limite permitido."
                    )
                destino.write(bloco)

                if progresso and tamanho_informado:
                    percentual = min(100, tamanho_total * 100 // tamanho_informado)
                    progresso(f"Baixando modelo local de português: {percentual}%")


def _extrair_modelo(arquivo, destino):
    with zipfile.ZipFile(arquivo) as pacote:
        membros = pacote.infolist()
        tamanho_descompactado = sum(membro.file_size for membro in membros)
        if tamanho_descompactado > MAX_MODEL_EXTRACTED_BYTES:
            raise AudioServiceError(
                "O modelo descompactado excede o limite permitido."
            )

        for membro in membros:
            caminho_zip = PurePosixPath(membro.filename)
            if (
                caminho_zip.is_absolute()
                or ".." in caminho_zip.parts
                or "\\" in membro.filename
                or not caminho_zip.parts
            ):
                raise AudioServiceError("O pacote do modelo contém um caminho inválido.")

            caminho_destino = destino.joinpath(*caminho_zip.parts)
            if membro.is_dir():
                caminho_destino.mkdir(parents=True, exist_ok=True)
                continue

            caminho_destino.parent.mkdir(parents=True, exist_ok=True)
            with pacote.open(membro) as origem, caminho_destino.open("wb") as saida:
                shutil.copyfileobj(origem, saida)


def _obter_modelo(progresso=None):
    diretorio_modelos = _diretorio_modelos()
    caminho_modelo = diretorio_modelos / MODEL_NAME
    if _modelo_valido(caminho_modelo):
        return caminho_modelo

    diretorio_modelos.mkdir(parents=True, exist_ok=True)
    if progresso:
        progresso("Baixando o modelo local de português (cerca de 31 MB)...")

    try:
        with tempfile.TemporaryDirectory(
            prefix="acolhemind-modelo-", dir=diretorio_modelos
        ) as temporario:
            pasta_temporaria = Path(temporario)
            arquivo_zip = pasta_temporaria / "modelo.zip"
            pasta_extracao = pasta_temporaria / "extraido"
            pasta_extracao.mkdir()

            _baixar_modelo(arquivo_zip, progresso)
            _extrair_modelo(arquivo_zip, pasta_extracao)

            modelo_extraido = pasta_extracao / MODEL_NAME
            if not _modelo_valido(modelo_extraido):
                raise AudioServiceError(
                    "O pacote baixado não contém um modelo de português válido."
                )
            shutil.move(str(modelo_extraido), str(caminho_modelo))
    except AudioServiceError:
        raise
    except (
        OSError,
        urllib.error.URLError,
        zipfile.BadZipFile,
        ValueError,
    ) as erro:
        raise AudioServiceError(
            f"Não foi possível preparar o modelo local: {erro}"
        ) from erro

    return caminho_modelo


def gravar_e_transcrever(duracao_segundos, progresso=None):
    if not isinstance(duracao_segundos, int) or not (
        1 <= duracao_segundos <= MAX_RECORDING_SECONDS
    ):
        raise AudioServiceError(
            f"A duração deve estar entre 1 e {MAX_RECORDING_SECONDS} segundos."
        )

    try:
        import sounddevice
        import vosk
    except ImportError as erro:
        raise AudioServiceError(
            "Dependências de áudio ausentes. Instale os pacotes de requirements.txt."
        ) from erro

    caminho_modelo = _obter_modelo(progresso)
    if progresso:
        progresso(f"Gravando pelo microfone por {duracao_segundos} segundos...")

    try:
        try:
            sounddevice.default.device = (5, 7)
        except Exception:
            pass
        audio = sounddevice.rec(
            int(duracao_segundos * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        sounddevice.wait()
    except Exception as erro:
        raise AudioServiceError(
            f"Não foi possível acessar o microfone: {erro}"
        ) from erro

    if progresso:
        progresso("Transcrevendo localmente...")

    try:
        vosk.SetLogLevel(-1)
        modelo = vosk.Model(str(caminho_modelo))
        reconhecedor = vosk.KaldiRecognizer(modelo, SAMPLE_RATE)
        dados_audio = audio.tobytes()
        for inicio in range(0, len(dados_audio), 4000):
            reconhecedor.AcceptWaveform(dados_audio[inicio : inicio + 4000])
        texto = json.loads(reconhecedor.FinalResult()).get("text", "").strip()
    except Exception as erro:
        raise AudioServiceError(
            f"Não foi possível transcrever o áudio localmente: {erro}"
        ) from erro

    if not texto:
        raise AudioServiceError(
            "Nenhuma fala foi reconhecida. Tente novamente em um ambiente mais silencioso."
        )

    return texto
