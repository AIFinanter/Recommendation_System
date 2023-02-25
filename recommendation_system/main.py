#! /usr/bin/python3
import math
import csv
import datetime
import heapq
import json
from tqdm import tqdm #tqdm是进度条库

def buildTarin():
    """
    处理数据集,
    users = 用户ID:{物品ID,评分}
    items = 物品ID:{用户ID,评分}
    itemUsers = 物品ID:[用户ID集]
    userItems = 用户ID:[物品ID集]
    allItems = 所有物品ID集
    :return:
    """

    startTime = datetime.datetime.now()
    users = dict()
    items = dict()
    itemUsers = dict()
    userItems = dict()
    allItems = set()

    with open('./ml-latest-small/ratings.csv') as f:
        f_csv = csv.reader(f)
        for row in f_csv:

            userId = row[0]
            #print("user id %d"% row[0])
            itemId = row[1]
            score = row[2]

            users.setdefault(userId,{})
            users[userId][itemId] = score

            userItems.setdefault(userId,[])
            userItems[userId].append(itemId)

            items.setdefault(itemId,{})
            items[itemId][userId] = score

            itemUsers.setdefault(itemId,[])
            itemUsers[itemId].append(userId)

            allItems.add(itemId)
    endTime = datetime.datetime.now()
    print('处理数据集耗时:{0}秒'.format((endTime-startTime).seconds))
    return users,userItems,items,itemUsers,sorted(allItems)

def avgDiffs(users,userItems,items,itemUsers,allItems):
    """
    计算两两物品评分差的均值
    itemABMatrix=物品IDA：{物品IDB:平均值}
    :param users:
    :param userItems:
    :param items:
    :param itemUsers:
    :param allItems:
    :return:
    """
    startTime = datetime.datetime.now()
    itemABMatrix = dict()
    allItemSize = len(allItems)
    pbar = tqdm(allItems)

    for itemIdA in pbar: #遍历所有物品，计算两两评分差的均值
        pbar.set_description("Processing %s"% itemIdA)
        itemABMatrix.setdefault(itemIdA,{})
        itemUsersA = itemUsers[itemIdA]
        for itemIdB in allItems:
            if itemIdA == itemIdB: #物品相同，不做处理
                continue
            itemUsersB = itemUsers[itemIdB]
            itemUsersAB = [x for x in itemUsersB if x in itemUsersA] #同时给物品A和物品B评分的用户集
            itemUsersABSize = len(itemUsersAB) #同时给物品AB评分的用户数
            if itemUsersABSize == 0: #没有同时给AB评分的用户，设为0
                itemABMatrix[itemIdA][itemIdB] = 0.0
                continue
            sum = 0.0 #差值的和
            avg = 0.0 #差值平均值
            for userId in itemUsersAB:#遍历同时给物品AB评分的用户，计算每个用户给AB物品评分的差值之和
                scoreA = float(users[userId][itemIdA])
                scoreB = float(users[userId][itemIdB])
                sum = sum + scoreA - scoreB
            avg = sum/itemUsersABSize
            itemABMatrix[itemIdA][itemIdB] = avg

    #print(json.dumps(itemABMatrix,sort_keys=False,indent=4,separators=(',',':'))
    endTime = datetime.datetime.now()
    print('计算评分差均值耗时:{0}秒'.format((endTime-startTime).seconds))
    return itemABMatrix

def recommendation(itemABMatrix,users,userItems,itemUsers,allItems,userId,N):
    """
    给用户推荐物品列表
    :param itemABMatrix:
    :param users:
    :param userItems:
    :param itemUsers:
    :param allItems:
    :param userId:
    :param N:
    :return:
    """
    startTime = datetime.datetime.now()
    pScore = dict()
    ratedItem = userItems[userId]
    notRatedItem = [x for x in allItems if x not in ratedItem] #用户未评过分的物品集
    #遍历用户未评过分的物品，计算预测分
    #公式:设,评过分的物品A、B 未评过分的物品C
    #[同时给AC评过分的人数*(用户对A的评分-物品AC评分差均值)+同时给BC评过分的人数*(用户对B的评分-物品BC评分差均值)]
    for notRatedItemId in notRatedItem:
        sum = 0.0
        avg = 0.0
        allUserSize = 0
        for ratedItemId in ratedItem:
            itemUsersA = itemUsers[ratedItemId]
            itemUsersB = itemUsers[notRatedItemId]
            itemUsersAB = [x for x in itemUsersB if x in itemUsersA] #同时给物品AB评分的用户数
            itemUsersABSize = len(itemUsersAB) #同时给物品AB评分的用户数

            allUserSize = allUserSize + itemUsersABSize
            sum = sum + itemUsersABSize*(users[userId][ratedItemId]-itemABMatrix[ratedItemId][notRatedItemId])
        pScore.setdefault(notRatedItemId,0.0)
        if allUserSize==0:#没有同时给物品评分的人，无法预测，不做处理
            continue

        avg = sum/allUserSize
        pScore[notRatedItemId] = avg
        endTime = datetime.datetime.now()

    print('推荐耗时:{0}秒'.format((endTime-startTime).seconds))
    return heapq.nlargest(N,pScore.items(),key=lambda x:x[1])



if __name__ == "__main__":
    startTime = datetime.datetime.now()
    users,userItems,items,itemUsers,allItems = buildTarin()
    #print(json.dumps(users,sort_keys=False,indent=4,separators=(',',':')))
    #print(json.dumps(userItems,sort_keys=False,indent=4,separators=(',',':')
    #print(json.dumps(items,sort_keys=False,indent=4,separators=(',',':'))
    #print(json.dumps(itemUsers,sort_keys=False,indent=4,separators=(',',':'))
    #print(json.dumps(allItems,sort_keys=False,indent=4,separators=(',',':')))
    itemABMatrix = avgDiffs(users,userItems,items,itemUsers,allItems)
    topN = recommendation(itemABMatrix,users,userItems,itemUsers,allItems,3,5)
    endTime = datetime.datetime.now()
    print('总耗时:{0}秒'.format((endTime-startTime).seconds))
    print(topN)