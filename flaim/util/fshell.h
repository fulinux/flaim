//-------------------------------------------------------------------------
// Desc:	Command line shell for FLAIM utilities - definitions.
// Tabs:	3
//
//		Copyright (c) 1999-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fshell.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FSHELL_HPP
#define FSHELL_HPP

#include "flaim.h"
#include "sharutil.h"
#include "fcs.h"
#include "fsv.h"

// Forward references

class FlmThreadContext;
class FlmParse;
class FlmCommand;

// Function typedefs

typedef RCODE (* THREAD_FUNC_p)(
				FlmThreadContext *		pThread,
				void *						pvAppData);

// Types of clipboard data

enum eClipboardDataType
{
	CLIPBOARD_EMPTY,
	CLIPBOARD_GEDCOM,
	CLIPBOARD_TEXT
};

typedef enum eClipboardDataType ClipboardDataType;

/*===========================================================================
Desc:		This class manages a clipboard
===========================================================================*/
class FlmClipboard : public F_Base
{
public:
	FlmClipboard( void);
	
	~FlmClipboard( void);

	RCODE putGedcom(
		NODE *			pTree);

	RCODE getGedcom(
		NODE * *			ppTreeRV,
		POOL *			pPool);

	RCODE putText(
		const char *	pszText);

	RCODE getText(
		char *			pszTextRV,
		FLMUINT *		puiSize);

	void clear( void);

	inline ClipboardDataType getType( void)
	{
		return( m_eDataType);
	}

private:

	POOL						m_pool;
	NODE *					m_pTree;
	char *					m_pszText;
	ClipboardDataType		m_eDataType;
};

/*===========================================================================
Desc:		This class manages a context or environment of variables.
===========================================================================*/
class FlmContext : public F_Base
{
public:
	FlmContext( void);

	~FlmContext( void);

	RCODE setup(
		FLMBOOL			bShared);

	RCODE setenv(
		const char *	pszVarName,
		const char *	pszVarValue);

	RCODE setenv(
		const char *	pszVarName,
		FLMUINT			uiVarValue);

	RCODE getenv(
		const char *	pszVarName,
		char *			pszVarValue,
		FLMUINT			uiMaxValLen);

	RCODE getenv(
		const char *	pszVarName,
		FLMUINT *		puiVarValue);

	RCODE setCurrDir( const char * pszCurrDir);

	RCODE getCurrDir( char * pszCurrDir);

	void lock( void);
	
	void unlock( void);

private:

	NODE * getVarNode(
		const char *	pszVarName);

	POOL						m_pool;
	NODE *					m_pEnvVarList;
	char						m_szCurrDir[ F_PATH_MAX_SIZE];
	FLMBOOL					m_bIsSetup;
	F_MUTEX					m_hMutex;		// Semaphore for controlling multi-thread
												// access.
};

/*===========================================================================
Desc:		This class manages the shared context for a group of threads.
===========================================================================*/

class FlmSharedContext : public FlmContext
{
public:
	FlmSharedContext( void);
	~FlmSharedContext( void);

	RCODE init(								// Initialized the share object.
		FlmSharedContext *	pSharedContext,
		FTX_INFO_p				pFtxInfo);

	RCODE setCacheSize( FLMUINT uiCacheSize);
	RCODE getCacheSize( FLMUINT * puiCacheSize);
	RCODE setCheckpointInterval( FLMUINT uiCheckpointInterval);
	RCODE getCheckpointInterval( FLMUINT * puiCheckpointInterval);
	inline FTX_INFO_p getFtxInfo( void) { return( m_pFtxInfo); }
	inline void setShutdownFlag( FLMBOOL * pbShutdownFlag) { m_pbShutdownFlag = pbShutdownFlag; }

	// Threads

	RCODE spawn(
		FlmThreadContext *	pThread,
		FLMUINT *				puiThreadID = NULL);	// ID of spawned thread

	RCODE spawn(
		const char *			pszThreadName,
		THREAD_FUNC_p			pFunc,
		void *					pvUserData,
		FLMUINT *				puiThreadID = NULL);	// ID of spawned thread

	void wait( void);

	void shutdown();		// Shutdown all threads in this shared context.

	RCODE killThread(
		FLMUINT					uiThreadID,
		FLMUINT					uiMaxWait = 0);

	RCODE setFocus( FLMUINT uiThreadID);

	FLMBOOL isThreadTerminating(
		FLMUINT					uiThreadID);

	RCODE getThread(
		FLMUINT					uiThreadID,
		FlmThreadContext **	ppThread);

	RCODE registerThread(
		FlmThreadContext *	pThread);

	RCODE deregisterThread(
		FlmThreadContext *	pThread);

	RCODE buildThreadList(
		void *					pPulldown);

	RCODE setThreadReturnCode( FLMUINT uiThreadID, RCODE rc);
	RCODE getThreadReturnCode( FLMUINT uiThreadID);

private:
	FlmSharedContext *	m_pParentContext;
	FLMBOOL					m_bPrivateShare;
	FTX_INFO_p				m_pFtxInfo;
	F_MUTEX					m_hMutex;	// Semaphore for controlling multi-thread
												// access.
	F_SEM						m_hSem;
	FlmThreadContext *	m_pThreadList;	// List of registered threads
	FLMBOOL					m_bLocalShutdownFlag;
	FLMBOOL *				m_pbShutdownFlag;
	FLMUINT					m_uiNextProcID;
	FlmVector				m_returnCodeHistory; //per-thread return codes, by id
};

/*===========================================================================
struct:	DB_CONTEXT
Desc:		This structure contains information for a particular database.
===========================================================================*/

typedef struct DBContextTag
{
	HFDB		hDb;
	FLMUINT	uiCurrContainer;
	FLMUINT	uiCurrIndex;
	FLMUINT	uiCurrDrn;
	FLMUINT	uiCurrSearchFlags;
} DB_CONTEXT;

/*===========================================================================
Desc:		This class manages a database context - FLAIM session and #N
			open databases.
===========================================================================*/

class FlmDbContext : public F_Base
{
#define MAX_DBCONTEXT_OPEN_DB		9
public:
	FlmDbContext( void);
	~FlmDbContext( void);

	inline FLMUINT getCurrDbId(
		void)
	{
		return m_uiCurrDbId;
	}

	inline void setCurrDbId(
		FLMUINT uiDbId)
	{
		if (uiDbId < MAX_DBCONTEXT_OPEN_DB)
		{
			m_uiCurrDbId = uiDbId;
		}
	}

	FLMBOOL getAvailDbId(
		FLMUINT *		puiDbId);

	FLMBOOL setDbHandle(
		FLMUINT	uiDbId,
		HFDB		hDb);
	
	HFDB getDbHandle(
		FLMUINT	uiDbId);

	FLMBOOL setCurrContainer(
		FLMUINT	uiDbId,
		FLMUINT	uiContainer);

	FLMUINT getCurrContainer(
		FLMUINT	uiDbId);

	FLMBOOL setCurrIndex(
		FLMUINT	uiDbId,
		FLMUINT	uiIndex);

	FLMUINT getCurrIndex(
		FLMUINT	uiDbId);

	FLMBOOL setCurrDrn(
		FLMUINT	uiDbId,
		FLMUINT	uiDrn);
	
	FLMUINT getCurrDrn(
		FLMUINT	uiDbId);

	FLMBOOL setCurrSearchFlags(
		FLMUINT	uiDbId,
		FLMUINT	uiSearchFlags);
	
	FLMUINT getCurrSearchFlags(
		FLMUINT	uiDbId);

	FLMBOOL setRecord(
		NODE *	pRecord);

	NODE * getRecord(
		POOL *	pPool);

	FLMBOOL setView(
		NODE *	pView);

	NODE * getView(
		POOL *	pPool);

	FLMBOOL setKey(
		NODE *	pKey);

	NODE * getKey(
		POOL *	pPool);

private:
	FLMUINT					m_uiCurrDbId;
	DB_CONTEXT				m_DbContexts [MAX_DBCONTEXT_OPEN_DB];
	POOL						m_pool;
	NODE *					m_pRecord;
	POOL						m_viewPool;
	NODE *					m_pView;
	POOL						m_keyPool;
	NODE *					m_pKey;
};

/*===========================================================================
Desc:		This class manages a thread.
===========================================================================*/
class FlmThreadContext : public F_Base
{
public:
	FlmThreadContext( void);
	virtual ~FlmThreadContext( void);

	RCODE setup(
		FlmSharedContext *	pSharedContext,
		const char *			pszThreadName,
		THREAD_FUNC_p			pFunc,
		void *					pvAppData);

	virtual RCODE execute( void);

	void shutdown();				// Needs to be thread-safe.

	RCODE setenv(					// Sets local context and optionally, global context.
		const char *	pszVarName,
		const char *	pszVarValue,
		FLMBOOL			bSetGlobalContext = FALSE);

	RCODE setenv(					// Sets local context and optionally, global context.
		const char *	pszVarName,
		FLMUINT			uiVarValue,
		FLMBOOL			bSetGlobalContext = FALSE);

	RCODE getenv(					// Look in local context first, global next.
		const char *	pszVarName,
		char *			pszVarValue,
		FLMUINT			uiMaxValLen,
		FLMBOOL			bSetLocal = FALSE);	// If retrieved from global context, set in local context.

	RCODE getenv(
		const char *	pszVarName,
		FLMUINT *		puiVarValue,
		FLMBOOL			bSetLocal = FALSE);

	inline FlmContext * getLocalContext( void)
	{
		return m_pLocalContext;
	}
	
	inline FlmSharedContext * getSharedContext( void)
	{
		return m_pSharedContext;
	}
	
	inline FLMBOOL * getShutdownFlagAddr( void)
	{
		return( &m_bShutdown);
	}
	
	inline void setShutdownFlag( void)
	{
		m_bShutdown = TRUE;
	}
	
	inline FLMBOOL getShutdownFlag( void)
	{ 
		if( m_pThread && m_pThread->getShutdownFlag())
		{
			m_bShutdown = TRUE;
		}

		return( m_bShutdown);
	}
	
	inline void setNext( FlmThreadContext * pNext) 
	{
		m_pNext = pNext;
	}
	
	inline void setPrev( FlmThreadContext * pPrev)
	{
		m_pPrev = pPrev;
	}

	inline FlmThreadContext * getNext( void)
	{
		return( m_pNext);
	}
	
	inline FlmThreadContext * getPrev( void)
	{
		return( m_pPrev);
	}

	inline void setID( FLMUINT uiID)
	{
		m_uiID = uiID;
	}
	
	inline FLMUINT getID( void)
	{
		return( m_uiID);
	}

	inline void setScreen( FTX_SCREEN * pScreen)
	{
		m_pScreen = pScreen;
	}
	
	inline FTX_SCREEN_p getScreen( void) 
	{
		return( m_pScreen);
	}
	
	inline void setWindow( FTX_WINDOW * pWindow)
	{
		m_pWindow = pWindow;
	}
	
	inline FTX_WINDOW_p getWindow( void)
	{
		return( m_pWindow);
	}

	inline void setFlmThread( F_Thread * pThread)
	{ 
		m_pThread = pThread;
	}

	inline F_Thread * getFlmThread( void)
	{ 
		return( m_pThread);
	}

	void getName( char * pszName, FLMBOOL bLocked = FALSE);

	RCODE exec( void);
	
	void lock( void);
	
	void unlock( void);

	FLMBOOL funcExited()
	{
		return m_bFuncExited;
	}
	
	inline void setFuncExited() 
	{
		m_bFuncExited = TRUE;
	}
	
	RCODE getFuncErrorCode()
	{
		flmAssert( this->funcExited());
		return m_FuncRC;
	}
		
protected:

	FTX_SCREEN_p			m_pScreen;
	FTX_WINDOW_p			m_pWindow;

private:

	FLMBOOL					m_bShutdown;
	FLMUINT					m_uiID;
	FlmContext *			m_pLocalContext;
	FlmSharedContext *	m_pSharedContext;
	FlmThreadContext *	m_pNext;
	FlmThreadContext *	m_pPrev;
	F_MUTEX					m_hMutex;
	F_Thread *				m_pThread;
	THREAD_FUNC_p			m_pThrdFunc;
	void *					m_pvAppData;
#define MAX_THREAD_NAME_LEN	64
	char						m_pszName[ MAX_THREAD_NAME_LEN + 1];
	FLMBOOL					m_bFuncExited;
	RCODE						m_FuncRC;
};

/*===========================================================================
Desc:		This class parses a command line
===========================================================================*/
class FlmParse : public F_Base
{
public:
	FlmParse( void);
	~FlmParse( void);

	void setString(
		const char *		pszString);

	char * getNextToken( void);

private:

	char				m_pszString[ 512];
	char				m_pszToken[ 512];
	char *			m_pszCurPos;
};

/*===========================================================================
Desc:		This class manages a command-line shell
===========================================================================*/
class FlmShell : public FlmThreadContext
{
public:
	FlmShell( void);
	~FlmShell( void);

	RCODE setup(
		FlmSharedContext *	pSharedContext);

	// Methods that are invoked by the command objects
	
	RCODE registerDatabase(
		HFDB				hDb,
		FLMUINT *		puiDbId);

	RCODE getDatabaseHandle(
		FLMUINT			uiDbId,
		HFDB *			phDb);

	RCODE deregisterDatabase(
		FLMUINT			uiDbId);

	RCODE con_printf(
		char *		pucFormat, ...);

	RCODE con_printf(
		NODE *			pRec);

	void setOutputPaging(
		FLMBOOL		bEnabled);

	FLMBOOL getAbortFlag( void);

	RCODE execute( void);

	RCODE addCmdHistory(
		const char *	pszCmd);

	RCODE getPrevCmd(
		char *	pszCmd);

	RCODE getNextCmd(
		char *	pszCmd);

	RCODE registerCmd(
		FlmCommand *	pCmd);

	inline FlmClipboard * getLocalClipboard( void) { return( m_pLocalClipboard); }

private:

#define MAX_SHELL_OPEN_DB				10
#define MAX_SHELL_HISTORY_ITEMS		10
#define MAX_REGISTERED_COMMANDS		50
#define MAX_CMD_LINE_LEN				256

	FlmSharedContext *	m_pSharedContext;
	FTX_WINDOW_p			m_pTitleWin;
	HFDB						m_DbList[ MAX_SHELL_OPEN_DB];
	POOL						m_HistPool;
	NODE *					m_pCmdHist;
	NODE *					m_pLastCmd;
	NODE *					m_pCmdHistPos;
	POOL						m_ArgPool;
	FLMINT					m_iCurrArgC;
	char **					m_ppCurrArgV;
	FLMINT					m_iLastCmdExitCode;
	FLMBOOL					m_bPagingEnabled;
	FlmCommand *			m_ppCmdList[ MAX_REGISTERED_COMMANDS];
	FlmClipboard *			m_pLocalClipboard;

	RCODE parseCmdLine(
		const char *		pszString);

	RCODE executeCmdLine( void);
	
	RCODE selectCmdLineFromList(
		char *				pszCmdLineRV);
};

/*===========================================================================
Desc:		This class is used by the shell to perform commands it has parsed.
===========================================================================*/
class FlmCommand : public F_Base
{
public:
	FlmCommand( void) { m_pszCmdName[ 0] = '\0'; }
	virtual ~FlmCommand( void) {}

	// Methods that must be implemented in classes that extend this class.

	virtual FLMINT execute(
		FLMINT			iArgC,
		const char **	ppszArgV,
		FlmShell *		pShell) = 0;

	virtual void displayHelp(
		FlmShell *	pShell) = 0;

	inline void setName( 
		const char *	pszName)
	{ 
		f_strcpy( m_pszCmdName, pszName);
	}
	
	inline const char * getName( void)
	{
		return( m_pszCmdName);
	}

private:

	char		m_pszCmdName[ 256];
};

/*===========================================================================
Desc:		This class implements the database open command
===========================================================================*/
class FlmDbOpenCommand : public FlmCommand
{
public:

	FlmDbOpenCommand( void)
	{
	}
	
	~FlmDbOpenCommand( void)
	{
	}

	FLMINT execute(
		FLMINT			iArgC,
		const char **	ppszArgV,
		FlmShell *		pShell);

	void displayHelp(
		FlmShell *		pShell);
private:
};

/*===========================================================================
Desc:		This class implements the database close command
===========================================================================*/
class FlmDbCloseCommand : public FlmCommand
{
public:
	FlmDbCloseCommand( void) {}
	~FlmDbCloseCommand( void) {}

	FLMINT execute(
		FLMINT			iArgC,
		const char **	ppszArgV,
		FlmShell *		pShell);

	void displayHelp(
		FlmShell *		pShell);
};

/*===========================================================================
Desc:		This class implements the set command (for env. variables)
===========================================================================*/
class FlmSetEnvCommand : public FlmCommand
{
public:
	FlmSetEnvCommand( void)
	{
	}
	
	~FlmSetEnvCommand( void)
	{
	}

	FLMINT execute(
		FLMINT			iArgC,
		const char **	ppszArgV,
		FlmShell *		pShell);

	void displayHelp(
		FlmShell *		pShell);
};

/*===========================================================================
Desc:		This class implements the rec command (for record retrieval, etc.)
===========================================================================*/
class FlmRecCommand : public FlmCommand
{
public:
	FlmRecCommand( void)
	{
	}
	
	~FlmRecCommand( void)
	{
	}

	FLMINT execute(
		FLMINT			iArgC,
		const char **	ppszArgV,
		FlmShell *		pShell);

	void displayHelp(
		FlmShell *	pShell);
};

/*===========================================================================
Desc:		This class implements the trans command (for transactions)
===========================================================================*/
class FlmTransCommand : public FlmCommand
{
public:

	FlmTransCommand( void)
	{
	}
	
	~FlmTransCommand( void)
	{
	}

	FLMINT execute(
		FLMINT			iArgC,
		const char **	ppszArgV,
		FlmShell *		pShell);

	void displayHelp(
		FlmShell *		pShell);
};

/*===========================================================================
Desc:		This class implements the check command (FlmDbCheck)
===========================================================================*/
class FlmCheckCommand : public FlmCommand
{
public:

	FlmCheckCommand( void)
	{
	}
	
	~FlmCheckCommand( void)
	{
	}

	FLMINT execute(
		FLMINT			iArgC,
		const char **	ppszArgV,
		FlmShell *		pShell);

	void displayHelp(
		FlmShell *		pShell);
};

/*===========================================================================
Desc:		This class implements the backup command (FlmDbBackup)
===========================================================================*/
class FlmBackupCommand : public FlmCommand
{
public:

	FlmBackupCommand( void)
	{
	}
	
	~FlmBackupCommand( void)
	{
	}

	FLMINT execute(
		FLMINT			iArgC,
		const char **	ppszArgV,
		FlmShell *		pShell);

	void displayHelp(
		FlmShell *		pShell);
};

/*===========================================================================
Desc:		This class implements the restore command (FlmDbRestore)
===========================================================================*/
class FlmRestoreCommand : public FlmCommand
{
public:
	FlmRestoreCommand( void)
	{
	}
	
	~FlmRestoreCommand( void)
	{
	}

	FLMINT execute(
		FLMINT			iArgC,
		const char **	ppszArgV,
		FlmShell *		pShell);

	void displayHelp(
		FlmShell *		pShell);
};

/*===========================================================================
Desc:
===========================================================================*/
class FlmDbConfigCommand : public FlmCommand
{
public:

	FlmDbConfigCommand( void)
	{
	}
	
	~FlmDbConfigCommand( void)
	{
	}

	FLMINT execute(
		FLMINT			iArgC,
		const char **	ppszArgV,
		FlmShell *		pShell);

	void displayHelp(
		FlmShell *		pShell);
};

/*===========================================================================
Desc:
===========================================================================*/
class FlmDbGetConfigCommand : public FlmCommand
{
public:
	FlmDbGetConfigCommand( void)
	{
	}
	
	~FlmDbGetConfigCommand( void)
	{
	}

	FLMINT execute(
		FLMINT			iArgC,
		const char **	ppszArgV,
		FlmShell *		pShell);

	void displayHelp(
		FlmShell *		pShell);
};

/*===========================================================================
Desc:		This class implements the sysinfo command
===========================================================================*/
class FlmSysInfoCommand : public FlmCommand
{
public:
	FlmSysInfoCommand( void)
	{
	}
	
	~FlmSysInfoCommand( void)
	{
	}

	FLMINT execute(
		FLMINT			iArgC,
		const char **	ppszArgV,
		FlmShell *		pShell);

	void displayHelp(
		FlmShell *		pShell);
};

/*===========================================================================
Desc:		This class implements various database tests
===========================================================================*/
class FlmTestCommand : public FlmCommand
{
public:

	FlmTestCommand( void)
	{
	}
	
	~FlmTestCommand( void)
	{
	}

	FLMINT execute(
		FLMINT			iArgC,
		const char **	ppszArgV,
		FlmShell *		pShell);

	void displayHelp(
		FlmShell *		pShell);
};

/*===========================================================================
Desc:		This class implements the database upgrade command
===========================================================================*/
class FlmDbUpgradeCommand : public FlmCommand
{
public:

	FlmDbUpgradeCommand( void)
	{
	}
	
	~FlmDbUpgradeCommand( void)
	{
	}

	FLMINT execute(
		FLMINT			iArgC,
		const char **	ppszArgV,
		FlmShell *		pShell);

	void displayHelp(
		FlmShell *		pShell);
};

/*===========================================================================
Desc:		Converts a file to a hex-equivalent ASCII file
===========================================================================*/
class  FlmHexConvertCommand : public FlmCommand
{
public:

	FlmHexConvertCommand( void)
	{
	}
	
	~FlmHexConvertCommand( void)
	{
	}

	FLMINT execute(
		FLMINT			iArgC,
		const char **	ppszArgV,
		FlmShell *		pShell);

	void displayHelp(
		FlmShell *		pShell);
};

/*===========================================================================
Desc:		Generates a FLAIM web authorization packet
===========================================================================*/
class FlmAuthPacketGenCommand : public FlmCommand
{
public:
	FlmAuthPacketGenCommand( void)
	{
	}
	
	~FlmAuthPacketGenCommand( void)
	{
	}

	FLMINT execute(
		FLMINT			iArgC,
		const char **	ppszArgV,
		FlmShell *		pShell);

	void displayHelp(
		FlmShell *		pShell);
};

/*===========================================================================
Desc:		This class implements the file copy command.
===========================================================================*/
class FlmCopyCommand : public FlmCommand
{
public:

	FlmCopyCommand( void);
	
	~FlmCopyCommand( void);

	FLMINT execute(
		FLMINT			iArgC,
		const char **	ppszArgV,
		FlmShell *		pShell);

	void displayHelp(
		FlmShell *		pShell);
};

/*===========================================================================
Desc:		This class implements the file delete command.
===========================================================================*/
class FlmDeleteCommand : public FlmCommand
{
public:

	FlmDeleteCommand( void)
	{
	}

	~FlmDeleteCommand( void)
	{
	}

	FLMINT execute(
		FLMINT			iArgC,
		const char **	ppszArgV,
		FlmShell *		pShell);

	void displayHelp(
		FlmShell *		pShell);
};

#endif
