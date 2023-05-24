# 유저 관련 서비스 클래스
# 유저 관련 비즈니스 로직을 처리하는 클래스

from App.model.User import User

class UserService:
    def __init__(self):
        self.userList = []

    def addUser(self, user):
        self.userList.append(user)

    def getUserList(self):
        return self.userList

    def getUser(self, userId):
        for user in self.userList:
            if user.getUserId() == userId:
                return user
        return None

    def getUserByMappingId(self, mappingId):
        for user in self.userList:
            if user.getMappingId() == mappingId:
                return user
        return None

    def updateUser(self, userId, mappingId):
        user = self.getUser(userId)
        if user is None:
            return False
        user.setMappingId(mappingId)
        return True

    def deleteUser(self, userId):
        user = self.getUser(userId)
        if user is None:
            return False
        self.userList.remove(user)
        return True

    def __str__(self):
        return "UserService(userList=" + str(self.userList) + ")"

