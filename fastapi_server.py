from fastapi import FastAPI, WebSocket
from track_0 import track_data, country_balls_amount
import numpy as np
import asyncio
import glob
from collections import Counter
import uvicorn


app = FastAPI(title='Tracker assignment')
imgs = glob.glob('imgs/*')
country_balls = [{'cb_id': x, 'img': imgs[x % len(imgs)]} for x in range(country_balls_amount)]
id_entrance = {}
print('Started')


def centroid_calc(bbox_value):
    """
    Ищем центр бокса
    """
    x_center = int((bbox_value[0] + bbox_value[2]) / 2.0)
    y_center = int((bbox_value[1] + bbox_value[3]) / 2.0)
    
    return (x_center, y_center)


def dist_calc(point_a, point_b):
    """
    Ищем расстояние между двумя точками (расст. Эвклид)
    """

    return (((point_a[0]-point_b[0])**2 + (point_a[1]-point_b[1])**2)**0.5)


def is_point_on_line(point_a, point_b, point_new, thr):
    '''
    Лежит ли новая точка на линии АБ
    '''
    k = (point_a[1] - point_b[1]) / (point_a[0] - point_b[0] + 0.000001)
    b = point_b[1] - k*point_b[0]
    y = k*point_new[0] + b

    y_min = (1 - thr) * y
    y_max = (1 + thr) * y

    if y_min < point_new[1] < y_max:
        answer = True
    else:
        answer = False

    return answer


def is_dist_correct(mean_dist, new_dist, thr):
    '''
    Равны ли примерно расстояния между (новой точкой + последней трека) и 
    средним расстоянием между точками для данного трека
    '''

    dist_min = (1 - thr) * mean_dist
    dist_max = (1 + thr) * mean_dist

    if dist_min < new_dist < dist_max:
        answer = True
    else:
        answer = False

    return answer


def tracker_soft(el, last_ids_balls_list, centroid_ids_dict, ids_balls_dist_dict):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов

    Ограничения:
    - необходимо использовать как можно меньше ресурсов (представьте, что
    вы используете embedded устройство, например Raspberri Pi 2/3).
    -значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме
    """
    # вначале присвоим по порядку айди
    if el['frame_id'] == 1:
        print('*********************first***********************')
        i = 0
        for x in el['data']:
            x['track_id'] = i
            last_ids_balls_list.append(i)
            i += 1
            if x['bounding_box']:
                # считаем для каждого ббокса центроиды и расстояния
                centroid_ids_dict[x['track_id']] = [centroid_calc(x['bounding_box'])]
                ids_balls_dist_dict[x['track_id']] = [0]
            else:
                # если не повезло и на старте детектор не смог кого-то найти
                centroid_ids_dict[x['track_id']] = []
                ids_balls_dist_dict[x['track_id']] = []

    # на втором фрейме тоже по порядку расставляем айди
    elif el['frame_id'] == 2:
        print('*********************second**********************')
        temp_ids_balls_list = []
        i = 0
        for x in el['data']:
            x['track_id'] = i
            temp_ids_balls_list.append(i)
            i += 1
            if x['bounding_box']:
                # считаем для каждого ббокса центр и расстояние
                centroid_ids_dict[x['track_id']].append(centroid_calc(x['bounding_box']))
                if len(centroid_ids_dict[x['track_id']]) > 1:
                    ids_balls_dist_dict[x['track_id']].append(dist_calc(centroid_ids_dict[x['track_id']][-2],
                                                                        centroid_ids_dict[x['track_id']][-1]))
                else:
                    ids_balls_dist_dict[x['track_id']].append(0)
            else:
                # если снова не везет с детектором, то пропуск
                pass

        last_ids_balls_list = temp_ids_balls_list

    # с 3 фрейма начинает работать основной алгоритм
    else:
        print(f"*********************frame={el['frame_id']}*********************")
        # создадим временные списки и словари для хранения данных на данном фрейме
        temp_ids_balls_list = []
        temp_centroid_ids_dict = {}
        temp_ids_balls_dist_dict = {}
        new_object_count = 1

        for x in el['data']:
            if x['bounding_box']:
                temp_point = centroid_calc(x['bounding_box'])
                k = 1
                # ищем на чьей прямой лежит данный центроид
                for id in centroid_ids_dict:
                    if len(centroid_ids_dict[id]) > 1:
                        point_on_line = is_point_on_line(centroid_ids_dict[id][-2],
                                                         centroid_ids_dict[id][-1],
                                                         temp_point,
                                                         0.1)
                        new_dist = dist_calc(temp_point,
                                             centroid_ids_dict[id][-1])
                        dist_correct = is_dist_correct(ids_balls_dist_dict[id][-1], new_dist, 0.15)

                        # если центроид лежит на прямой и совпадает по дистанции, то
                        # это известный объект
                        if point_on_line and dist_correct:
                            x['track_id'] = id

                            centroid_ids_dict[id].append(temp_point)
                            ids_balls_dist_dict[id].append(new_dist)
                            temp_ids_balls_list.append(id)
                        else:
                            # если центроид никому не подошел, значит новый объект
                            if k == len(centroid_ids_dict):
                                print('new_object_count', new_object_count)
                                new_id = max(centroid_ids_dict) + 1 * new_object_count
                                x['track_id'] = new_id

                                # записываем во времянный словарь, так как нельзя в
                                # основной по нему итерируемся
                                temp_centroid_ids_dict[new_id] = [temp_point]
                                temp_ids_balls_dist_dict[new_id] = [0]
                                temp_ids_balls_list.append(new_id)
                                new_object_count += 1
                            else:
                                k += 1
                    else:
                        # если новый объект с прошлого фрейма и у него нет "истории",
                        # тогда сравниваем по расстоянию. Если расстояние меньше 70 (примерно подобрал),
                        # то это новый объект с прошлого фрейма, иначе новый
                        new_dist = dist_calc(temp_point, centroid_ids_dict[id][-1])
                        if new_dist < 70:
                            x['track_id'] = id

                            centroid_ids_dict[id].append(temp_point)
                            ids_balls_dist_dict[id].append(new_dist)
                            temp_ids_balls_list.append(id)
                        else:
                            # если никому не подошел, значит новый объект
                            if k == len(centroid_ids_dict):
                                print('new_object_count', new_object_count)
                                new_id = max(centroid_ids_dict) + 1 * new_object_count
                                x['track_id'] = new_id

                                # записываем во времянный словарь, так как нельзя в
                                # основной по нему итерируемся
                                temp_centroid_ids_dict[new_id] = [temp_point]
                                temp_ids_balls_dist_dict[new_id] = [0]
                                temp_ids_balls_list.append(new_id)
                                new_object_count += 1
                            else:
                                k += 1

            else:
                pass

        centroid_ids_dict = {**centroid_ids_dict, **temp_centroid_ids_dict}
        ids_balls_dist_dict = {**ids_balls_dist_dict, **temp_ids_balls_dist_dict}
        last_ids_balls_list = temp_ids_balls_list

    # не продумана идея, если объект возвращается, к примеру на фреймах -1 и -3 он есть,
    # а на фрейме -2 его детектор пропустил, можно было бы обыграть двойную дистанцию и
    # больший дрейф прямой, или еще что-то
    return el, last_ids_balls_list, centroid_ids_dict, ids_balls_dist_dict


def tracker_strong(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов, скриншоты прогона

    Ограничения:
    - вы можете использовать любые доступные подходы, за исключением
    откровенно читерных, как например захардкодить заранее правильные значения
    'track_id' и т.п.
    - значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме

    P.S.: если вам нужны сами фреймы, измените в index.html значение make_screenshot
    на true для первого прогона, на повторном прогоне можете читать фреймы из папки
    и по координатам вырезать необходимые регионы.
    TODO: Ужасный костыль, на следующий поток поправить
    """
    return el


def aggregate_data(el):
    global id_entrance
    for x in el['data']:
        if x['cb_id'] in id_entrance:
            id_entrance[x['cb_id']].append(x['track_id'])
        else:
            id_entrance[x['cb_id']] = [x['track_id']]

    return el


def calc_tracker_metrics(id_entrance):
    right_choice = 0
    total_len = 0
    for k, v in id_entrance.items():
        occurence_count = Counter(v)
        max_occur_value, amount_of_entrance = occurence_count.most_common(1)[0] # value, amount of entrance
        # нужно считать подряд идущих, а не по всему списку.  Или брать топ-2
        if max_occur_value is None:
            # amount_of_entrance = 0
            max_occur_value, amount_of_entrance = occurence_count.most_common(2)[1]
        right_choice += amount_of_entrance
        total_len += len(v)
        print(f'Track: {k=} {max_occur_value=} {amount_of_entrance=} {amount_of_entrance/len(v):.02f}')
    print(f'Overall: {right_choice/total_len:.02f}')

    return id_entrance


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print('Accepting client connection...')
    centroid_ids_dict = {}
    last_ids_balls_list = []
    ids_balls_dist_dict = {}
    real_ids_and_track_ids = {}
    await websocket.accept()
    # отправка служебной информации для инициализации объектов
    # класса CountryBall на фронте
    await websocket.send_text(str(country_balls))
    for el in track_data:
        await asyncio.sleep(0.5)
        # TODO: part 1
        el, last_ids_balls_list, centroid_ids_dict, ids_balls_dist_dict = tracker_soft(el,
                                                                                       last_ids_balls_list,
                                                                                       centroid_ids_dict,
                                                                                       ids_balls_dist_dict)
        # TODO: part 2
        # el = tracker_strong(el)
        # отправка информации по фрейму
        await websocket.send_json(el)
        aggregate_data(el)
    calc_tracker_metrics(id_entrance)
    print('Bye..')

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8001)
