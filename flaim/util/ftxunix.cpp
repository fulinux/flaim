//-------------------------------------------------------------------------
// Desc:	Unix text user interface APIs - windowing.
// Tabs:	3
//
//		Copyright (c) 1997,1999-2003,2005-2006 Novell, Inc. All Rights Reserved.
//
//		This program is free software; you can redistribute it and/or
//		modify it under the terms of version 2 of the GNU General Public
//		License as published by the Free Software Foundation.
//
//		This program is distributed in the hope that it will be useful,
//		but WITHOUT ANY WARRANTY; without even the implied warranty of
//		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//		GNU General Public License for more details.
//
//		You should have received a copy of the GNU General Public License
//		along with this program; if not, contact Novell, Inc.
//
//		To contact Novell about this file by physical or electronic mail,
//		you may find current contact information at www.novell.com
//
// $Id: ftxunix.cpp 12328 2006-01-19 16:39:54 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaim.h"
#include "ftx.h"

#ifdef FLM_UNIX

#if defined( FLM_HPUX) || defined( FLM_OSF)
	#ifndef _XOPEN_CURSES
		#define _XOPEN_CURSES
	#endif
	#define _XOPEN_SOURCE_EXTENDED 1
#endif

/* curses.h pollutes name spaces like crazy; these crazy definitions
 * are required to get the code to compile cleanly on all platforms.
 */
#if defined( bool)
	#undef bool
#endif

#if defined( EO)
	#undef EO						// Compaq Tru64 has the strangest problems
#endif

#if defined( ERR)
	#undef ERR
#endif

#if defined( FLM_SOLARIS) 
	#define _WIDEC_H
#endif

#include <curses.h>

#ifdef FLM_AIX
	#ifdef wgetch
	#undef wgetch
	#endif
	extern "C"
	{
		extern int wgetch( WINDOW *);
		extern int clear( void);
	}
#endif

static int ungetChar;

/* Curses gives us only a limited number of color pairs. We use this
   static color_pairs array for only the colors we need. flm2curses is 
   used to convert from flaim colors to curses colors and last_pair
   is the last_pair is the last pair that we used. */

static short flm2curses[8];
static short color_pairs[8][8];
short last_pair = 0;

void 
	ftxUnixDisplayInit( void)
{
	initscr();
	noecho();
	cbreak();
	halfdelay(4);
	meta(stdscr, TRUE); 
	keypad(stdscr, TRUE);
	scrollok(stdscr, FALSE);
	move(0, 0);
	refresh();
	ungetChar = (int)ERR;
	if (has_colors())
	{
		start_color();

		/* initialize the mapping array */
		flm2curses[WPS_BLACK]     = COLOR_BLACK;
		flm2curses[WPS_BLUE]      = COLOR_BLUE;
		flm2curses[WPS_GREEN]     = COLOR_GREEN; 
		flm2curses[WPS_CYAN]      = COLOR_CYAN;
		flm2curses[WPS_RED]       = COLOR_RED;
		flm2curses[WPS_MAGENTA]   = COLOR_MAGENTA;
		flm2curses[WPS_BROWN]     = COLOR_YELLOW;
		flm2curses[WPS_LIGHTGRAY] = COLOR_WHITE;

		/* set default background */
		int defaultbg = A_NORMAL | ' ';
		bkgd(defaultbg); 
	}
}


void 
	ftxUnixDisplayFree( void)
{
	endwin();
}

void ftxUnixDisplayGetSize( FLMUINT * puiNumColsRV, FLMUINT * puiNumRowsRV)
{
	*puiNumColsRV = (FLMUINT)COLS;
	*puiNumRowsRV = (FLMUINT)LINES;
}

static int flm_to_curses_attr(
	int		attr)
{
	int fg, bg, curses_attr = 0;

	fg = attr & 0x0f;				  /* get foreground color */
	bg = (attr >> 4) & 0x07;

	curses_attr = (fg > WPS_LIGHTGRAY) ? A_BOLD : 0;
	fg &= 0x07;						  /* we have only the "dark" color now */
	if (has_colors())
	{
		if (color_pairs[bg][fg] == 0) /* not allocated yet */
		{
			if (last_pair >= COLOR_PAIRS) /* exhausted color pairs */
				color_pairs[bg][fg] = 1; /* the first allocated pair */
			else
			{
				color_pairs[bg][fg] = ++last_pair;
				init_pair(last_pair, flm2curses[fg], flm2curses[bg]);
			}
		}
		curses_attr |= COLOR_PAIR(color_pairs[bg][fg]);
	} 
	else
	{
		/* FTX applications have a blue background by default; we
		 *	use a reverse video attribute for any other background.
		 *	This works well for terminals that don't support color.
		 */
		curses_attr |= (bg != WPS_BLUE) ? A_REVERSE : 0;
	}

	return curses_attr;
}

void 
	ftxUnixDisplayChar(
		FLMUINT		uiChar,
		FLMUINT		uiAttr)
{
	addch( (int)uiChar | flm_to_curses_attr( (int)uiAttr));
}


void 
	ftxUnixDisplayRefresh( void)
{
	refresh();
}


void 
	ftxUnixDisplayReset( void)
{
	clearok( stdscr, TRUE);
	refresh();
}


void 
	ftxUnixDisplaySetCursorPos( 
		FLMUINT		uiCol,
		FLMUINT		uiRow
	)
{
	move( (int)uiRow, (int)uiCol);
	refresh();
}

/****************************************************************************
Name:	ftxUnixSimulatedKey
Desc:	simulate Ctrl + Shift + character keys
****************************************************************************/
static FLMUINT32 ftxUnixSimulatedKey(
	FLMUINT32		c)
{
	/* We simulate Insert, Delete, Home, End, PgUp and PgDn. We can
		also simulate the CTRL- combinations of these if needed */

	chtype ch = (chtype) c + 'a' - 1;	// make the switch readable
	switch (ch)
	{
	case 'i':
		c = WPK_INSERT;
		break;

	case 'd':						  
		c = WPK_DELETE;
		break;

	case 'b':						  // back
		c = WPK_PGUP;
		break;

	case 'f':						  // forward
		c = WPK_PGDN;
		break;

	case 'h':
		c = WPK_HOME;
		break;

	case 'e':
		c = WPK_END;
		break;

	default:
		c = (FLMUINT32)ERR;
		break;
	}
	return c;
}

static FLMUINT32 ftxUnixHandleEsc()
{
	/* 
		On unix ESC is the prefix for many function keys. It's a bad
		idea to use ESC as a character by itself because it can result
		in a delay of as much as a second.  If we don't handle all
		escape sequences, the first escape character can cause FLAIM to
		exit!  So, we discard unrecognized escape sequences here.
	*/

	int c = WPK_ESCAPE;
	int c2;
	
	if ((c2 = getch()) == ERR)
	{
		goto Exit;
	}

	switch( c2)
	{
		//simulate F1 via Esc-F1, etc.
		case '1':
			c = WPK_F1;
			break;
		
		case '2':
			c = WPK_F2;
			break;

		case '3':
			c = WPK_F3;
			break;

		case '4':
			c = WPK_F4;
			break;

		case '5':
			c = WPK_F5;
			break;

		case '6':
			c = WPK_F6;
			break;

		case '7':
			c = WPK_F7;
			break;

		case '8':
			c = WPK_F8;
			break;

		case '9':
			c = WPK_F9;
			break;

		case '0':
			c = WPK_F10;
			break;

		case 'i':
			c = WPK_INSERT;
			break;

		case 'd':
			c = WPK_DELETE;
			break;
			
			/* It's not possible to generate CTRL-LEFT, SHIFT-TAB etc in
				curses. So we used escape followed by the key to generate
				the missing WPK_* codes. This seems to work on Linux. */

		case KEY_F(0):			  /* Curses is sometimes very weird */
			c = WPK_ALT_F10;
			break;

		case '\t':
			c = WPK_STAB;		  
			break;

		case KEY_FIND:
		case KEY_HOME:
			c = WPK_CTRL_HOME;
			break;

		case KEY_END:
		case KEY_SELECT:
		case KEY_LL:
			c = WPK_CTRL_END;
			break;

		case KEY_LEFT:
			c = WPK_CTRL_LEFT;
			break;

		case KEY_RIGHT:
			c = WPK_CTRL_RIGHT;
			break;

		case KEY_DOWN:
			c = WPK_CTRL_DOWN;
			break;

		case KEY_UP:
			c = WPK_CTRL_UP;
			break;

		case 0x000A:
		case 0x000D:
		case KEY_ENTER:
			c = WPK_CTRL_ENTER;
			break;

		case KEY_NPAGE:
			c = WPK_CTRL_PGDN;
			break;

		case KEY_PPAGE:
			c = WPK_CTRL_PGUP;
			break;

		case KEY_IC:
			c = WPK_CTRL_INSERT;
			break;

		default:
		{
			if (c2 >= '0' && c2 <= '9')
			{
				c = WPK_ALT_0 + c2 - '0';
			} 
			else if (c2 >= 'a' && c2 <= 'z')
			{
				c = WPK_ALT_A + c2 - 'a';
			}
			else if ((c2 >= 1) && (c <= 032))
			{
				c = ftxUnixSimulatedKey( c2); /* Ctrl + Shift + character */
			}
			else if (c2 >= KEY_F(1) && c2 <= KEY_F(10))
			{
				c = WPK_ALT_F1 + c - KEY_F( 1);
			} 
			else if (c2 == erasechar() || c2 == '' || c2 == 0127)
			{
				c = WPK_ESCAPE;		  /* Escape followed by Erase or DEL */
			} 
			else if (c2 == 033)
			{
				// Treat a double escape as WPK_ESCAPE
				c = WPK_ESCAPE;  
				break;
			}
			else
			{
				// discard unrecognized escape sequence
				c = ERR;
				while (getch() != ERR)
					;
			}
			break;
		}
	}
Exit:
	return c;
}

/* On Unix some terminal types (notably Solaris xterm) do not generate
	proper key codes for Insert, Home, PgUp, PgDn etc. Use a different
	terminal emulator (rxvt for eg). They can also be simulated by the
	key combination Meta-Shift-I, Meta-Shift-U, Meta-Shift-D etc. */

FLMUINT 
	ftxUnixKBGetChar( void)
{
	int c;

 Again:
	if (ungetChar != ERR)
	{
		c = ungetChar;
		ungetChar = ERR;
	}
	else
	{
		while ((c = getch()) == ERR);
	}

	if (c == killchar())
	{
		c = WPK_DELETE;
	}
	else if (c == erasechar())
	{
		c = WPK_BACKSPACE;
	}
	else if (c == '\t')
	{
		c = WPK_TAB;
	}
	else if( c >= 1 && c <= 032 && c != 10 && c != 13)
		c = WPK_CTRL_A + (c - 1);
	else if ((c >= (128 + '0')) && (c <= (128 + '9')))
		c = WPK_ALT_0 + (c - 128 - '0'); /* Alt + Number */
	else if ((c >= (128 + 'a')) && (c <= (128 + 'z')))
		c = WPK_ALT_A + (c - 128 - 'a'); /* Alt + character */
	else if ((c >= 128) && (c <= (128 + 032)))
		c = ftxUnixSimulatedKey(c - 128); /* Ctrl + Shift + character */
	else if (c >= KEY_F(1) && c <= KEY_F(10))
		c = WPK_F1 + c - KEY_F(1); /* Function key */
	else if (c >= KEY_F(11) && c <= KEY_F(20))
		c = WPK_SF1 + c - KEY_F(11); /* shift Function key */
	else
	{
		switch( c)
		{
		case KEY_F(0):				  /* Curses is sometimes very weird */
			c = WPK_ALT_F10;
			break;

		case KEY_BACKSPACE:
			c = WPK_BACKSPACE;
			break;

		case 033:				/* Escape Character */
		{
			c = ftxUnixHandleEsc();
			break;
		}
		case 0127:					  /* DEL Character */
		case KEY_DC:
			c = WPK_DELETE;
			break;

		case KEY_FIND:
		case KEY_HOME:
			c = WPK_HOME;
			break;

		case KEY_END:
		case KEY_SELECT:
		case KEY_LL:
			c = WPK_END;
			break;

		case KEY_LEFT:
			c = WPK_LEFT;
			break;

		case KEY_RIGHT:
			c = WPK_RIGHT;
			break;

		case KEY_DOWN:
			c = WPK_DOWN;
			break;

		case KEY_UP:
			c = WPK_UP;
			break;

		case 0x000A:
		case 0x000D:
		case KEY_ENTER:
			c = WPK_ENTER;
			break;

		case KEY_NPAGE:
			c = WPK_PGDN;
			break;

		case KEY_PPAGE:
			c = WPK_PGUP;
			break;

		case KEY_IC:
			c = WPK_INSERT;
			break;
		}
	}

	if (c == ERR)
		goto Again;					  // discarded ESC prefix 

	return((FLMUINT)c);
}


FLMBOOL 
	ftxUnixKBTest()
{
	int c;

	if (ungetChar != ERR)
	{
		c = ungetChar;
	}
	else
	{
		if ((c = getch()) != ERR)
		{
			ungetChar = c;
		}
	}
	return((c == ERR) ? FALSE : TRUE);
}

#else
	#if defined( FLM_NLM) && !defined( __MWERKS__)
		void ftxunix_dummy_func()
		{
		}
	#endif
#endif // FLM_UNIX
