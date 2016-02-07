#ifndef __VIZIA_DEFINES_H__
#define __VIZIA_DEFINES_H__

namespace Vizia{

    enum Mode {
        PLAYER,             // synchronous player mode
        SPECTATOR,          // synchronous spectator mode
        ASYNC_PLAYER,       // asynchronous player mode
        ASYNC_SPECTATOR,    // asynchronous spectator mode
    };

    enum ScreenFormat {
        CRCGCB = 0,
        CRCGCBDB = 1,
        RGB24 = 2,
        RGBA32 = 3,
        ARGB32 = 4,
        CBCGCR = 5,
        CBCGCRDB = 6,
        BGR24 = 7,
        BGRA32 = 8,
        ABGR32 = 9,
        GRAY8 = 10,
        DEPTH_BUFFER8 = 11,
        DOOM_256_COLORS8 = 12,
    };

    enum ScreenResolution {
        RES_40X30,
        RES_60X45,
        RES_80X50,      // 16:10
        RES_80X60,
        RES_100X75,
        RES_120X75,     // 16:10
        RES_120X90,
        RES_160X100,
        RES_160X120,
        RES_200X120,
        RES_200X150,
        RES_240X135,
        RES_240X150,
        RES_240X180,
        RES_256X144,
        RES_256X160,
        RES_256X192,
        RES_320X200,
        RES_320X240,
        RES_400X225,	// 16:9
        RES_400X300,
        RES_480X270,	// 16:9
        RES_480X360,
        RES_512X288,	// 16:9
        RES_512X384,
        RES_640X360,	// 16:9
        RES_640X400,
        RES_640X480,
        RES_720X480,	// 16:10
        RES_720X540,
        RES_800X450,	// 16:9
        RES_800X480,
        RES_800X500,	// 16:10
        RES_800X600,
        RES_848X480,	// 16:9
        RES_960X600,	// 16:10
        RES_960X720,
        RES_1024X576,	// 16:9
        RES_1024X600,	// 17:10
        RES_1024X640,	// 16:10
        RES_1024X768,
        RES_1088X612,	// 16:9
        RES_1152X648,	// 16:9
        RES_1152X720,	// 16:10
        RES_1152X864,
        RES_1280X720,	// 16:9
        RES_1280X854,
        RES_1280X800,	// 16:10
        RES_1280X960,
        RES_1280X1024,	// 5:4
        RES_1360X768,	// 16:9
        RES_1366X768,
        RES_1400X787,	// 16:9
        RES_1400X875,	// 16:10
        RES_1400X1050,
        RES_1440X900,
        RES_1440X960,
        RES_1440X1080,
        RES_1600X900,	// 16:9
        RES_1600X1000,	// 16:10
        RES_1600X1200,
        RES_1680X1050,	// 16:10
        RES_1920X1080,  // 16:9
        RES_1920X1200,
        RES_2048X1536,
        RES_2560X1440,
        RES_2560X1600,
        RES_2560X2048,
        RES_2880X1800,
        RES_3200X1800,
        RES_3840X2160,
        RES_3840X2400,
        RES_4096X2160,
        RES_5120X2880,
    };

    enum GameVariable {
        KILLCOUNT,
        ITEMCOUNT,
        SECRETCOUNT,
        FRAGCOUNT,
        HEALTH,
        ARMOR,
        DEAD,
        ON_GROUND,
        ATTACK_READY,
        ALTATTACK_READY,
        SELECTED_WEAPON,
        SELECTED_WEAPON_AMMO,
        AMMO0,
        AMMO1,
        AMMO2,
        AMMO3,
        AMMO4,
        AMMO5,
        AMMO6,
        AMMO7,
        AMMO8,
        AMMO9,
        WEAPON0,
        WEAPON1,
        WEAPON2,
        WEAPON3,
        WEAPON4,
        WEAPON5,
        WEAPON6,
        WEAPON7,
        WEAPON8,
        WEAPON9,
        USER1,
        USER2,
        USER3,
        USER4,
        USER5,
        USER6,
        USER7,
        USER8,
        USER9,
        USER10,
        USER11,
        USER12,
        USER13,
        USER14,
        USER15,
        USER16,
        USER17,
        USER18,
        USER19,
        USER20,
        USER21,
        USER22,
        USER23,
        USER24,
        USER25,
        USER26,
        USER27,
        USER28,
        USER29,
        USER30,
    };

    static const int UserVariablesNumber = 30;
    static const int SlotsNumber = 10;

    enum Button {
        ATTACK = 0,
        USE = 1,
        JUMP = 2,
        CROUCH = 3,
        TURN180 = 4,
        ALTATTACK = 5,
        RELOAD = 6,
        ZOOM = 7,

        SPEED = 8,
        STRAFE = 9,

        MOVE_RIGHT = 10,
        MOVE_LEFT = 11,
        MOVE_BACKWARD = 12,
        MOVE_FORWARD = 13,
        TURN_RIGHT = 14,
        TURN_LEFT = 15,
        LOOK_UP = 16,
        LOOK_DOWN = 17,
        MOVE_UP = 18,
        MOVE_DOWN = 19,
        LAND = 20,
        //SHOWSCORES 20

        SELECT_WEAPON1 = 21,
        SELECT_WEAPON2 = 22,
        SELECT_WEAPON3 = 23,
        SELECT_WEAPON4 = 24,
        SELECT_WEAPON5 = 25,
        SELECT_WEAPON6 = 26,
        SELECT_WEAPON7 = 27,
        SELECT_WEAPON8 = 28,
        SELECT_WEAPON9 = 29,
        SELECT_WEAPON0 = 30,

        SELECT_NEXT_WEAPON = 31,
        SELECT_PREV_WEAPON = 32,
        DROP_SELECTED_WEAPON = 33,

        ACTIVATE_SELECTED_ITEM = 34,
        SELECT_NEXT_ITEM = 35,
        SELECT_PREV_ITEM = 36,
        DROP_SELECTED_ITEM = 37,

        LOOK_UP_DOWN_DELTA = 38,
        TURN_LEFT_RIGHT_DELTA = 39,
        MOVE_FORWARD_BACKWARD_DELTA = 40,
        MOVE_LEFT_RIGHT_DELTA = 41,
        MOVE_UP_DOWN_DELTA = 42,
    };

    static const int BinaryButtonsNumber = 38;
    static const int DeltaButtonsNumber = 5;
    static const int ButtonsNumber = 43;

}
#endif
