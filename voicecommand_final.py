from __future__ import division
import re
import sys
import threading
import pyautogui
import pyperclip
import os
import speech_recognition as sr
from pywinauto import Application
import time
import openai
import win32com.client as wincl



#발급 받은 API 키 설정
OPENAI_API_KEY = "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"

# openai API 키 인증
openai.api_key = OPENAI_API_KEY

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\quswj\Downloads\voicecommand-382709-2b72b5f3cefe.json"

from google.cloud import speech
import pyaudio
from six.moves import queue

# Audio recording parameters
RATE = 20000  ## 음성인식 성능향상
CHUNK = int(RATE / 30)  # 100ms ## 음성인식 성능향상


class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)

def type_text(text):
    pyperclip.copy(text)
    pyautogui.hotkey('ctrl', 'v')



# 기존 임포트 및 설정 코드 생략 ...

def speak(text, rate=1.0):
    speak = wincl.Dispatch("SAPI.SpVoice")
    speak.Rate = rate
    speak.Speak(text)

waiting = False

def waiting_message():
    global waiting
    while waiting:
        print("잠시만 기다려 주세요...")
        speak("잠시만 기다려 주세요...")
        time.sleep(1)

def is_windows_spoken():

    global waiting
    # 음성 인식 객체 생성
    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        r.adjust_for_ambient_noise(source)

    messages = []
    while True:
        print("Listening...")
        speak("음성인식 실행")
        with mic as source:
            r.non_blocking = True
            audio = r.listen(source)

        try:
            text = r.recognize_google_cloud(audio, language='ko-KR')
            if "윈도우" in text:
                speak("네 말씀하세요.")
                print("명령어를 말씀해주세요.")
                return True
            elif "입력" in text:
                speak("잠시만 기다려주세요")
                print("잠시만 기다려주세요")
                return False
            #gpt api 사용.

            elif "검색" in text:
                print("검색 질문을 말씀해주세요.")
                speak("검색 질문을 말씀해주세요.")
                with mic as source:
                    audio = r.listen(source)

                waiting = True
                waiting_thread = threading.Thread(target=waiting_message)
                waiting_thread.start()

                question = r.recognize_google_cloud(audio, language='ko-KR')
                messages.append({"role": "user", "content": question})

                answer = speech_to_gpt(messages)

                waiting = False
                waiting_thread.join()

                print(f'GPT: {answer}')
                messages.append({"role": "assistant", "content": answer})
                type_text(answer)
                #speak(answer)

            elif "종료" in text:
                speak("종료하겠습니다.")
                sys.exit("Exiting")
            else:
                speak("다시 말씀해주세요.")
                print("다시 말씀해주세요.")
        except sr.UnknownValueError:
            speak("다시 말씀해주세요.")
            print("다시 말씀해주세요.")
        except sr.WaitTimeoutError:
            # 타임아웃 에러 발생 시 다시 음성 수집 시작
            continue


def listen_print_loop(responses):
    transcript = ""
    final_transcript = ""

    for response in responses:
        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        # 기존에 출력된 텍스트 지우기
        sys.stdout.write("\033[F\033[K")

        # 최종 결과만 출력하고 나머지는 저장
        if result.is_final:
            final_transcript += transcript + " "
            print(final_transcript)

            if re.search(r'\b(음성인식 꺼 줘|quit)\b', final_transcript, re.I):
                speak("종료하겠습니다.")
                sys.exit("Exiting")

            if re.search(r'\b(음성인식 종료|quit)\b', final_transcript, re.I):
                speak("종료하겠습니다.")
                sys.exit("Exiting")

            if "메모장" in final_transcript:
                app = Application().start("notepad.exe")
                time.sleep(0.5)
                break

            if "크롬" in final_transcript:
                app = Application().start("C:\Program Files\Google\Chrome\Application\chrome.exe")
                time.sleep(0.5)
                break

            if "한글" in final_transcript:
                app = Application().start("C:\Program Files (x86)\HNC\Office 2022\HOffice120\Bin\Hwp.exe")
                time.sleep(0.5)
                break

            if "검색" in final_transcript:
                pyautogui.keyDown("ctrl")
                pyautogui.press("f")
                pyautogui.keyUp("ctrl")
                time.sleep(0.5)
                break

            if "복사" in final_transcript:
                pyautogui.keyDown("ctrl")
                pyautogui.press("c")
                pyautogui.keyUp("ctrl")
                time.sleep(0.5)
                break

            if "붙여 넣기" in final_transcript:
                pyautogui.keyDown("ctrl")
                pyautogui.press("v")
                pyautogui.keyUp("ctrl")
                time.sleep(0.5)
                break

            if "뒤로 가기" in final_transcript:
                pyautogui.keyDown("ctrl")
                pyautogui.press("z")
                pyautogui.keyUp("ctrl")
                time.sleep(0.5)
                break

            if "닫기" in final_transcript:
                pyautogui.keyDown("alt")
                pyautogui.press("f4")
                pyautogui.keyUp("alt")
                time.sleep(0.5)
                break

            if "다음" in final_transcript:
                pyautogui.press("space")
                time.sleep(0.5)
                break

            if "나가기" in final_transcript:
                pyautogui.press("esc")
                time.sleep(0.5)
                break

            if "엔터" in final_transcript:
                pyautogui.press("enter")
                time.sleep(0.5)
                break

            # ------------------------------------------5/7
            if "한글" in transcript:
                pyautogui.press("hangul")
                time.sleep(0.5)
                break

            if "영어" in transcript:
                pyautogui.press("hangul")
                time.sleep(0.5)
                break

            if "저장" in transcript:
                pyautogui.hotkey('ctrl', 's')
                time.sleep(0.5)
                break

            if "전체 선택" in transcript:
                pyautogui.hotkey('ctrl', 'a')
                time.sleep(0.5)
                break

            if "캡처" in transcript:
                pyautogui.hotkey('shift', 'win', 's')
                time.sleep(0.5)
                break

            if "캡스락" in transcript:
                pyautogui.press("capslocck")
                time.sleep(0.5)
                break

            if "자르기" in transcript:
                pyautogui.hotkey('ctrl', 'x')
                time.sleep(0.5)
                break

            # ------------------------------------------5/13

            if "새로고침" in transcript:
                pyautogui.press("f5")
                time.sleep(0.5)
                break

            if "시작" in transcript:
                pyautogui.press("f5")
                time.sleep(0.5)
                break

            if "바탕화면" in transcript:
                pyautogui.hotkey('win', 'd')
                time.sleep(0.5)
                break

            if "최대화" in transcript:
                pyautogui.hotkey('win', 'up')
                time.sleep(0.5)
                break

            if "삭제" in transcript:
                pyautogui.press("delete")
                time.sleep(0.5)
                break

            if "실행" in transcript:
                pyautogui.hotkey('ctrl', 'alt', 'f10')
                time.sleep(0.5)
                break

            if "작업 관리자" in transcript:
                pyautogui.hotkey('ctrl', 'shift', 'esc')
                time.sleep(0.5)
                break

            final_transcript = ""
        else:
            sys.stdout.write(transcript + '\r')
            sys.stdout.flush()

    return final_transcript


def speech_to_text():
    # create a recognizer instance
    r = sr.Recognizer()
    # use the system default microphone as the audio source
    with sr.Microphone() as source:
        print("Speak:")
        # adjust for ambient noise
        r.adjust_for_ambient_noise(source)
        # listen for audio
        audio = r.listen(source)
    try:
        # recognize speech using Google Speech Recognition
        text = r.recognize_google(audio, language='ko-KR')
        speak("입력하겠습니다.")
        print("You said: " + text)
        return text
    except sr.WaitTimeoutError:
        print("Listening timed out. Please try again.")
        return None
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return None

def speech_to_gpt(messages):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    chat_response = completion.choices[0].message.content
    return chat_response



def main():
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = 'ko-KR'  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code)
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    while True:
        windows_spoken = is_windows_spoken()
        if windows_spoken:
            end_time = time.time() + 5  # 5 seconds
            while time.time() < end_time:
                with MicrophoneStream(RATE, CHUNK) as stream:
                    audio_generator = stream.generator()
                    requests = (speech.StreamingRecognizeRequest(audio_content=content)
                                for content in audio_generator)
                    responses = client.streaming_recognize(streaming_config, requests)
                    listen_print_loop(responses)
                    break

        elif not windows_spoken:
            command = speech_to_text()
            if command:
                type_text(command)
        else:
            time.sleep(0.05)


if __name__ == '__main__':
    main()

# [END speech_transcribe_streaming_mic]
