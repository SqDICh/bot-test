import logging
import torch
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from transformers import AutoModelForCausalLM, AutoTokenizer

# === Настройки ===
API_TOKEN = "7429610790:AAGuxxBWVER9luO-ajki7Ljgk5tgrUnXLfc"  # Вставь свой токен от @BotFather
MODEL_NAME = "sberbank-ai/rugpt3small_based_on_gpt2"

# === Логирование ===
logging.basicConfig(level=logging.INFO)

# === Инициализация бота ===
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# === Загрузка модели ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# === Функция генерации предсказаний ===
def generate_prediction():
    prompt = "Твое предсказание: "
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(input_ids, max_length=30, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.9)

    prediction = tokenizer.decode(output[0], skip_special_tokens=True)
    return prediction.replace(prompt, "").strip()

# === Обработчик команды /start ===
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Привет! Отправь любое сообщение, и я пришлю тебе предсказание.")

# === Обработчик сообщений ===
@dp.message_handler()
async def send_prediction(message: types.Message):
    prediction = generate_prediction()
    await message.reply(f"✨ {prediction} ✨")

# === Запуск бота ===
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
