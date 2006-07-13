//------------------------------------------------------------------------------
// Desc:	FLAIM's utility routines for presenting selection and statistics lists
//
// Tabs:	3
//
//		Copyright (c) 2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: flm_lutl.h 3117 2006-01-19 13:34:36 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

RCODE FLMAPI flstIndexManagerThread(
	IF_Thread *		pThread);

RCODE FLMAPI flstMemoryManagerThread(
	IF_Thread *		pThread);
