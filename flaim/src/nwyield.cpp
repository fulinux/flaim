//-------------------------------------------------------------------------
// Desc:	Special CPU yielding routines for Netware
// Tabs:	3
//
//		Copyright (c) 2000-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: nwyield.cpp 12315 2006-01-19 15:16:37 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#ifdef FLM_NLM

/****************************************************************************/
/* Procedures imported from the Kernel */
/****************************************************************************/

extern "C"
{

void StopBell(void);

unsigned long kQueCount(
	unsigned long	uiQueHandle);

unsigned long CEvaluateExpression(
	unsigned char **	commandLine,
	unsigned long *	stackFrame,
	unsigned long *	number);

/****************************************************************************
	Data items imported from the kernel
****************************************************************************/

extern unsigned long	CpuCurrentProcessor;
extern unsigned long	WorkToDoListHead;

/****************************************************************************/

struct OldPerCpuStruct
{
	unsigned long reserved0[24];

	/* offset 0x60 */
	unsigned long reserved1[3];
	unsigned long PSD_ThreadStartClocks;
	unsigned long reserved2[4];

	unsigned long reserved3[40];

	/* offset 0x120 */
	unsigned long PSD_LocalWTDHead;
};

/****************************************************************************/
#if !defined( __MWERKS__)
extern unsigned long ReadInternalClock(void);
#else
unsigned long ReadInternalClock(void);
#endif

// Local function prototypes

FSTATIC int LocalYieldInit( void);

FSTATIC void OldOSCompatibleYield(void);


}	// extern "C"

#if !defined( __MWERKS__)
#pragma aux ReadInternalClock = \
0x0F 0x31                         	/* RDTSC  				*/\
modify exact [EAX EDX];
#else
unsigned long ReadInternalClock(void)
{
	__asm
	{
		rdtsc
		ret
	}
}
#endif

/****************************************************************************/

static unsigned long					yieldTimeSlice = 0;
static unsigned long					stackFrame[0x100] = {0};
static unsigned long *				ThreadFeederQue = 0;
static struct OldPerCpuStruct *	pPerCpuDataArea = 0;

/****************************************************************************
Desc:	This API is called once the first time NWYieldIfTime() is called if
		kYieldIfTimeSliceUp isn't supported by the host OS.
****************************************************************************/
FSTATIC int LocalYieldInit(void)
{
	unsigned char *	buffer;
	unsigned char *	pointer;
	unsigned long		ccode;
	unsigned long		address;
	unsigned long		(*ConvertMicroSecondsToClocks)(unsigned long data);

	// Fixup ThreadFeededQue

	buffer = (unsigned char *)"SERVER.NLM|ThreadFeederQue";

	// Get the address from the debugger

	ccode = CEvaluateExpression( &buffer, &stackFrame[0], &address);
	if (ccode != 0)
	{
		return( -1);
	}

	ThreadFeederQue = (unsigned long *)address;

	// Setup yieldTimeSlice
	// First get a pointer to ConvertMicroSecondsToClocks and then verify it

	pointer = (unsigned char *)(&StopBell);
	pointer = pointer + 0x1ea;
	if (pointer[0] != 0x8b ||
			pointer[1] != 0x44 ||
			pointer[2] != 0x24 ||
			pointer[3] != 0x04 ||
			pointer[4] != 0x33 ||
			pointer[5] != 0xd2 ||
			pointer[6] != 0xf7)
	{
		return( -1);
	}

	*(unsigned long *)(&ConvertMicroSecondsToClocks) = (unsigned long)pointer;

	// Now calculate our desired time slice in clocks

	yieldTimeSlice = (*ConvertMicroSecondsToClocks)(200);

	// Get a pointer to the per-CPU data area

	buffer = (unsigned char *)"LOADER.NLM|PerCpuDataArea";
	ccode = CEvaluateExpression( &buffer, &stackFrame[0], &address);
	if (ccode != 0)
	{
		return( -1);
	}

	pPerCpuDataArea = (struct OldPerCpuStruct *)address;
	return( 0);
}

/****************************************************************************
Desc:	This routine does everything that kYieldIfTimeSliceUp() does, and it is
		compatible with 5.0 SP4 and 5.1 release.  It shouldn't be used on host
		OSs that support kYieldIfTimeSliceUp().
****************************************************************************/
FSTATIC void OldOSCompatibleYield(void)
{
	unsigned long	timeStamp;
	unsigned long	threadStartTime;

	if ((CpuCurrentProcessor == 0) && (WorkToDoListHead != 0))
	{

		// If we are P0 and there is legacy WTD waiting, then yield

		kYieldThread();
		return;
	}

	if ((pPerCpuDataArea->PSD_LocalWTDHead != 0) ||
		 (kQueCount( ThreadFeederQue[CpuCurrentProcessor]) != 0))
	{

		// If there is MPK WTD or a feederQ thread waiting, then yield

		kYieldThread();
		return;
	}

	timeStamp = ReadInternalClock();
	threadStartTime = pPerCpuDataArea->PSD_ThreadStartClocks;

	// Note 32 bit arithmetic is sufficient and less overhead

	if ((timeStamp - threadStartTime) > yieldTimeSlice)
	{
		kYieldThread();
	}
}

/****************************************************************************
Desc: This is the routine to call for time sensitive yielding.  The first
		time it is called, it will change itself to a JMP to the appropriate
		yield procedure.  If it is on 5.1 SP1 or greater, it will jump to
		kYieldIfTimeSliceUp().  If it is on 5.0 SP4 or 5.1 release, it will
		jump to OldOSCompatibleYield().  Otherwise it will jump to
		kYieldThread().
****************************************************************************/
void NWYieldIfTime(void)
{
	unsigned char *	fixup;
	unsigned long		address;

	*(unsigned long *)(&address) = ImportPublicSymbol(
			(unsigned long)f_getNLMHandle(),
			(unsigned char *)"\x13" "kYieldIfTimeSliceUp");

	fixup = (unsigned char *)(&NWYieldIfTime);

	if (address == 0)
	{
		// We couldn't import the procedure, so see if our local routine 
		// is compatible.

		if (LocalYieldInit() == 0)
		{
			address = (unsigned long)(&OldOSCompatibleYield);
		}
		else
		{
			address = (unsigned long)(&kYieldThread);
		}
	}

	fixup[0] = 0xE9;
	++fixup;
	address = address - (4 + (unsigned long)fixup);
	*(unsigned long *)fixup = address;
}

#endif	// FLM_NLM

/****************************************************************************
Desc:
****************************************************************************/
#if defined( FLM_OSX)
void gv_nwyield()
{
}
#endif
