//-------------------------------------------------------------------------
// Desc:	Utility routines for presenting selection and statistics lists - definitions.
// Tabs:	3
//
//		Copyright (c) 2000-2001,2003-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flm_lutl.h 12212 2006-01-19 14:00:20 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flm_dlst.h"

RCODE flstIndexManagerThread(
	F_Thread *		pThread);

RCODE flstMemoryManagerThread(
	F_Thread *		pThread);

RCODE flstTrackerMonitorThread(
	F_Thread *		pThread);
