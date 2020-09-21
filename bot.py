import json
import logging
import torch
from random import randint
from RNN import init_rnn, evaluate
from secret import token
from telegram import ReplyKeyboardMarkup
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters,
                          ConversationHandler)

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

CHOICE, THEME, SEND = range(3)

reply_keyboard = [['Карточный текст', 'Рандомный текст'],
                  ['Текст для описания', 'Настроенный'],
                  ['Ничего']]
markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)

# load model
with open('meta', 'r') as meta_data:
    meta = json.load(meta_data)
model, seq, index_char, char_index = init_rnn(meta['file_name'], meta['model']['hidden_size'],
                                              meta['model']['embedding_size'], meta['model']['n_layers'])
model.load_state_dict(torch.load('model'))


def start(update, context):
    update.message.reply_text("Привет!")
    update.message.reply_text("Я Lorem Newspom Bot, я умею генерировать всякий бред из актуальных новостей, "
                              "для использования в верстке, или где нибуь еще🤔")
    update.message.reply_text("Для начала выбери вариант, который нужен", reply_markup=markup)
    return CHOICE


def card_text(update, context):
    update.message.reply_text(evaluate(model, char_index, index_char, " ", 120, 0.2))
    return ConversationHandler.END


def random_text(update, context):
    update.message.reply_text(evaluate(model, char_index, index_char, " ", randint(100, 10000)))
    return ConversationHandler.END


def big_text(update, context):
    update.message.reply_text(evaluate(model, char_index, index_char, " ", 1200))
    return ConversationHandler.END


def settings(update, context):
    update.message.reply_text("Введите размер текста")
    return THEME


def theme(update, context):
    context.user_data['scale'] = update.message.text
    update.message.reply_text("Введите тему текста. (Можно просто пробел)")
    return SEND


def get_setting_text(update, context):
    update.message.reply_text(evaluate(model, char_index, index_char,
                                       update.message.text, int(context.user_data['scale'].strip())))
    return ConversationHandler.END


def done(update, context):
    update.message.reply_text("Ничего, так ничего 🤷‍")
    return ConversationHandler.END


def main():
    updater = Updater(token, use_context=True)
    dp = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start),
                      MessageHandler(Filters.regex('^Карточный текст$'),
                                     card_text),
                      MessageHandler(Filters.regex('^Рандомный текст$'),
                                     random_text),
                      MessageHandler(Filters.regex('^Текст для описания$'),
                                     big_text),
                      MessageHandler(Filters.regex('^Настроенный$'),
                                     settings)
                      ],

        states={
            CHOICE: [
                    MessageHandler(Filters.regex('^Карточный текст$'),
                                   card_text),
                    MessageHandler(Filters.regex('^Рандомный текст$'),
                                   random_text),
                    MessageHandler(Filters.regex('^Текст для описания$'),
                                   big_text),
                    MessageHandler(Filters.regex('^Настроенный$'),
                                   settings)
                    ],
            THEME: [MessageHandler(Filters.text, theme)],
            SEND: [MessageHandler(Filters.text, get_setting_text)]
        },

        fallbacks=[MessageHandler(Filters.regex('^Ничего$'), done)]
    )

    dp.add_handler(conv_handler)
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
